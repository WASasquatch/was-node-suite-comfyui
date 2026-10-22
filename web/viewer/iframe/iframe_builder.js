/**
 * Iframe Builder - Builds HTML content for the viewer iframe
 */

import { escapeHtml } from "../utils/helpers.js";
import {
  detectContentType,
  renderContent,
  getViewStyles,
  getViewScripts,
  getView,
  viewUsesBaseStyles
} from "../views/view_loader.js";
import { getPrismTheme, getPrismScriptTags } from "../views/code_scripts.js";
import { loadCssFile, getCssBasePath } from "../utils/css.js";
import { getFullTheme, themeToCssVars } from "../utils/theme.js";
import { frameGuard, placeContent } from "./frame_guard.js";

const LIST_SEPARATOR = "\n---LIST_SEPARATOR---\n";

/** What a frame shows above content it took scripts, handlers, forms or frames out of. */
const INERT_NOTE = "Scripts and forms in this content are switched off. To run them, set "
  + "viewer.allow_scripts to true in config.yaml, restart ComfyUI and run this node again. "
  + "Only do that for workflows you trust.";

// Cached CSS content
let cachedBaseCss = null;
let cachedListCss = null;

/**
 * Initialize CSS by loading from files
 */
async function initCss() {
  if (cachedBaseCss === null) {
    const basePath = getCssBasePath();
    cachedBaseCss = await loadCssFile(`${basePath}/iframe-base.css`);
    cachedListCss = await loadCssFile(`${basePath}/iframe-list.css`);
  }
}

// Start loading CSS immediately
initCss();

/**
 * Get base CSS with theme variables
 * @param {object} theme - Basic theme tokens (bg, fg, border, accent)
 * @returns {string} CSS styles
 */
function getBaseStyles(theme) {
  // Get full theme for comprehensive CSS variables
  const fullTheme = getFullTheme();
  const cssVars = themeToCssVars(fullTheme);

  if (cachedBaseCss) {
    return cssVars + cachedBaseCss;
  }
  // Fallback inline styles if CSS not loaded yet
  return cssVars + `
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 4px; background: var(--theme-bg); color: var(--theme-fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.6; overflow-x: hidden; word-wrap: break-word; }
    a { color: var(--theme-accent); }
    pre { background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px; overflow-x: hidden; white-space: pre-wrap; word-wrap: break-word; font-family: "Fira Code", "Consolas", "Monaco", monospace; font-size: 13px; }
    code { background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 4px; font-family: "Fira Code", "Consolas", "Monaco", monospace; }
    pre code { background: transparent; padding: 0; }
    blockquote { border-left: 4px solid var(--theme-accent); margin: 12px 0; padding: 8px 16px; background: rgba(0,0,0,0.15); }
    img { max-width: 100%; height: auto; border-radius: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; border: 1px solid var(--theme-border); }
    th, td { border: 1px solid var(--theme-border); padding: 8px; text-align: left; }
    th { background: rgba(0,0,0,0.2); }
    hr { border: none; border-top: 1px solid var(--theme-border); margin: 16px 0; }
    h1, h2, h3, h4, h5, h6 { margin-top: 16px; margin-bottom: 8px; color: var(--theme-fg); }
    h1:first-child, h2:first-child, h3:first-child, h4:first-child, h5:first-child, h6:first-child { margin-top: 0; }
    ul, ol { padding-left: 24px; margin: 8px 0; }
    li { margin: 2px 0; }
  `;
}

/**
 * Get list CSS
 * @returns {string} CSS styles
 */
function getListStyles() {
  if (cachedListCss) {
    return cachedListCss;
  }
  // Fallback inline styles if CSS not loaded yet
  return `
    .list-item { background: rgba(0,0,0,0.2); border: 1px solid var(--theme-border); border-radius: 6px; padding: 12px; margin-bottom: 8px; position: relative; }
    .list-item:last-child { margin-bottom: 0; }
    .list-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .list-header-left { display: flex; align-items: center; gap: 8px; }
    .list-checkbox { width: 16px; height: 16px; cursor: pointer; accent-color: var(--theme-accent); }
    .list-index { display: inline-block; background: var(--theme-accent); color: #fff; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; }
    .list-item.excluded { opacity: 0.5; }
    .list-item.excluded .list-index { background: #666; }
    .copy-btn { background: transparent; border: 1px solid var(--theme-border); border-radius: 4px; color: var(--theme-fg); cursor: pointer; padding: 4px 8px; font-size: 11px; display: flex; align-items: center; gap: 4px; transition: all 0.2s; }
    .copy-btn:hover { background: var(--theme-accent); color: #fff; border-color: var(--theme-accent); }
    .copy-btn.copied { background: #22c55e; border-color: #22c55e; color: #fff; }
    .list-content { white-space: pre-wrap; word-wrap: break-word; }
  `;
}

/**
 * A value written as a javascript literal inside a script element.
 * @param {unknown} value - Anything JSON can hold.
 * @returns {string} Its JSON, with every `<` as an escape.
 */
function scriptJson(value) {
  return JSON.stringify(value)
    .replace(/</g, "\\u003c");
}

/**
 * Text as base64 of its UTF-8 bytes.
 * @param {string} text
 * @returns {string} Base64 holding nothing but letters, digits, `+`, `/` and `=`.
 */
function encodeText(text) {
  const bytes = new TextEncoder().encode(String(text ?? ""));
  let binary = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(start, start + 0x8000));
  }
  return btoa(binary);
}

/**
 * A fresh value for a frame's script nonce.
 * @returns {string} 32 hex digits from the browser's random source.
 */
function makeNonce() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * The policy a frame carries while `viewer.allow_scripts` is off.
 * @param {string} nonce - The nonce the frame's own scripts carry.
 * @returns {string} A content policy running only scripts carrying `nonce`.
 */
function inertPolicy(nonce) {
  return "default-src 'self' data: blob:; img-src 'self' data: blob:; media-src 'self' data: blob:; "
    + "font-src 'self' data:; style-src 'self' 'unsafe-inline' data:; "
    + `script-src 'nonce-${nonce}'; script-src-attr 'none'; connect-src 'self' data: blob:; `
    + "frame-src 'none'; child-src 'none'; worker-src 'none'; object-src 'none'; "
    + "base-uri 'none'; form-action 'none'";
}

/**
 * Script tags the pack wrote, each given the frame's nonce.
 * @param {string} html - Script tags from the pack's own view files.
 * @param {string} nonce
 * @returns {string} The same tags carrying `nonce`.
 */
function withNonce(html, nonce) {
  return String(html || "").replace(/<script(?=[\s>])/gi, `<script nonce="${nonce}"`);
}

/**
 * A call of one of the frame's own functions, as script text.
 * @param {Function} fn - A function that reads nothing outside its own body.
 * @param {...unknown} args - Its arguments, each something JSON can hold.
 * @returns {string} Javascript calling `fn` with `args`.
 */
function frameCall(fn, ...args) {
  return `(${fn.toString()})(${args.map(scriptJson).join(", ")});`;
}

/**
 * Build the list items' markup, each with its keep box and copy button.
 * @param {string[]} items - The items' text.
 * @param {object} theme - Theme tokens
 * @param {number[]} excluded - Excluded item indices
 * @returns {string} HTML content
 */
function buildListMarkup(items, theme, excluded) {
  return items.map((item, idx) => {
    const itemType = detectContentType(item);
    let itemHtml = "";
    if (itemType === "html") {
      itemHtml = item;
    } else if (itemType === "markdown") {
      itemHtml = renderContent(item, "markdown", theme);
    } else {
      itemHtml = `<div class="list-content">${escapeHtml(item)}</div>`;
    }
    const isExcluded = excluded.includes(idx);
    return `<div class="list-item${isExcluded ? ' excluded' : ''}" data-idx="${idx}">
      <div class="list-header">
        <div class="list-header-left">
          <input type="checkbox" class="list-checkbox" data-idx="${idx}" ${isExcluded ? '' : 'checked'}>
          <span class="list-index">${idx + 1} / ${items.length}</span>
        </div>
        <button class="copy-btn" data-index="${idx}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          Copy
        </button>
      </div>
      ${itemHtml}
    </div>`;
  }).join("");
}

/**
 * The script answering the list's keep boxes and copy buttons.
 * @param {string[]} items - The items' text, which the copy buttons copy.
 * @param {string|number} nodeId - The node the toggles are reported for.
 * @returns {string} One script tag.
 */
function buildListScript(items, nodeId) {
  return `<script>
    (function () {
      var binary = atob(${scriptJson(encodeText(JSON.stringify(items)))});
      var bytes = new Uint8Array(binary.length);
      for (var b = 0; b < binary.length; b++) bytes[b] = binary.charCodeAt(b);
      var items = JSON.parse(new TextDecoder().decode(bytes));
      var nodeId = ${scriptJson(String(nodeId ?? ""))};
      function copyItem(idx) {
        const text = items[idx];
        navigator.clipboard.writeText(text).then(() => {
          const btn = document.querySelector('.copy-btn[data-index="' + idx + '"]');
          if (!btn) return;
          btn.classList.add('copied');
          btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
          setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';
          }, 1500);
        });
      }
      function toggleItem(idx, checked) {
        const item = document.querySelector('.list-item[data-idx="' + idx + '"]');
        if (item) item.classList.toggle('excluded', !checked);
        window.parent.postMessage({ type: 'was-viewer-toggle', idx: idx, checked: checked, nodeId: nodeId }, '*');
      }
      document.addEventListener('change', function (event) {
        var box = event.target;
        if (!box || !box.matches || !box.matches('input.list-checkbox[data-idx]')) return;
        var idx = parseInt(box.getAttribute('data-idx'), 10);
        if (Number.isInteger(idx) && idx >= 0 && idx < items.length) toggleItem(idx, box.checked);
      });
      document.addEventListener('click', function (event) {
        var btn = event.target && event.target.closest ? event.target.closest('button.copy-btn[data-index]') : null;
        if (!btn) return;
        var idx = parseInt(btn.getAttribute('data-index'), 10);
        if (Number.isInteger(idx) && idx >= 0 && idx < items.length) copyItem(idx);
      });
    })();
  <\/script>`;
}

// The policy every framed document carries while `viewer.allow_scripts` is on.
const POLICY = "default-src 'self' data: blob:; img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self' data:; style-src 'self' 'unsafe-inline' data:; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; connect-src 'self' data: blob:; frame-src 'none'; object-src 'none'; form-action 'none'";

const POLICY_TAG = `<meta http-equiv="Content-Security-Policy" content="${POLICY}">`;

/**
 * Build complete iframe HTML content
 * @param {string} content - Content to render
 * @param {string} contentType - Detected content type
 * @param {object} theme - Theme tokens
 * @param {number[]} [excluded=[]] - Excluded item indices for list content
 * @param {string|number} [nodeId=''] - Node ID for message routing
 * @param {{allowScripts?: boolean}} [options] - `allowScripts: true`, which only the server's
 *   `viewer.allow_scripts` may set, lets the content run its own scripts. Otherwise the content
 *   is placed with nothing in it that runs or navigates, and only the frame's own scripts run.
 * @returns {string} Complete HTML document
 */
export function buildIframeContent(content, contentType, theme, excluded = [], nodeId = '', options = {}) {
  const allowScripts = options?.allowScripts === true;
  content = String(content ?? "");

  // Only include base styles if the view wants them
  const useBase = viewUsesBaseStyles(contentType);
  const baseStyles = useBase ? getBaseStyles(theme) : themeToCssVars(getFullTheme());
  const listStyles = useBase ? getListStyles() : "";

  const isListContent = content.includes(LIST_SEPARATOR);
  const htmlView = getView("html");
  const fullDocument = !isListContent && contentType === "html"
    && Boolean(htmlView?.isFullDocument?.(content));

  if (fullDocument && allowScripts) {
    // The theme goes in first, at the lowest specificity the page can carry, so a
    // document that styles itself paints over it and one that does not reads as the
    // node around it.
    // The policy leads the document. Anything after it is covered, including markup that
    // opens a script before it opens a head.
    return `${POLICY_TAG}<style id="was-theme-base">${baseStyles}</style>${content}`;
  }

  let bodyContent = "";
  let listScript = "";

  if (fullDocument) {
    bodyContent = content;
  } else if (isListContent) {
    const items = content.split(LIST_SEPARATOR);
    bodyContent = buildListMarkup(items, theme, excluded);
    listScript = buildListScript(items, nodeId);
  } else {
    bodyContent = renderContent(content, contentType, theme);
  }

  const nonce = allowScripts ? "" : makeNonce();
  const view = getView(contentType);
  const viewStyles = fullDocument ? "" : getViewStyles(contentType, theme);
  const needsPrism = !fullDocument && (view?.needsPrism?.() ||
                     contentType === "markdown" && (content.includes("```") || content.includes("<code")));
  const prismTheme = needsPrism ? getPrismTheme(theme) : "";
  const prismScripts = needsPrism ? getPrismScriptTags(nonce) : "";

  let katexStyles = "";
  if (contentType === "markdown") {
    const mdView = getView("markdown");
    if (mdView && mdView.hasLatex && mdView.hasLatex(content)) {
      katexStyles = `<style>${mdView.getKatexCss?.() || ""}</style>`;
    }
  }

  // Get inline scripts from view (for views that use getScripts() instead of getScriptData())
  const viewScripts = fullDocument ? "" : getViewScripts(contentType, content);

  if (!allowScripts) {
    const doctype = fullDocument && !/^\s*<!doctype/i.test(content) ? "" : "<!DOCTYPE html>\n";
    const styles = fullDocument
      ? `<style id="was-theme-base">${baseStyles}</style>`
      : `<style>${baseStyles}${listStyles}${viewStyles}${prismTheme}</style>`;
    return `${doctype}<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="${inertPolicy(nonce)}">
  <meta http-equiv="x-dns-prefetch-control" content="off">
  <script nonce="${nonce}">${frameCall(frameGuard, { nodeId: String(nodeId ?? "") })}<\/script>
  ${styles}
  ${katexStyles}
  ${prismScripts}
</head>
<body><script nonce="${nonce}">${frameCall(placeContent, encodeText(bodyContent), { fullDocument, note: INERT_NOTE })}<\/script>${withNonce(listScript, nonce)}${withNonce(viewScripts, nonce)}</body>
</html>`;
  }

  // Bootstrap script for postMessage-based script injection
  // This avoids embedding large scripts inline which can break HTML parsing
  const bootstrapScript = `<script>
    window.WAS_NODE_ID = ${scriptJson(String(nodeId || ''))};
    window.WAS_SCRIPTS_LOADED = false;
    window.addEventListener('message', function(event) {
      if (event.source !== window.parent) return;
      if (event.data && event.data.type === 'was-theme-update') {
        try {
          var cssVars = event.data.cssVars || '';
          var styleEl = document.getElementById('was-theme-vars');
          if (!styleEl) {
            styleEl = document.createElement('style');
            styleEl.id = 'was-theme-vars';
            document.head.appendChild(styleEl);
          }
          styleEl.textContent = cssVars;
        } catch (e) {
          console.error('[WAS Viewer] Theme update error:', e);
        }
        return;
      }

      if (event.data && event.data.type === 'was-inject-scripts' && !window.WAS_SCRIPTS_LOADED) {
        window.WAS_SCRIPTS_LOADED = true;
        var scripts = event.data.scripts || [];
        var initQueue = [];

        function loadNext(index) {
          if (index >= scripts.length) {
            // All scripts loaded, run init functions
            initQueue.forEach(function(initCode) {
              try { new Function(initCode)(); } catch(e) { console.error('[WAS Viewer] Init error:', e); }
            });
            return;
          }
          var scriptData = scripts[index];
          try {
            new Function(scriptData.code)();
            if (scriptData.init) {
              initQueue.push(scriptData.init);
            }
            loadNext(index + 1);
          } catch(e) {
            console.error('[WAS Viewer] Script injection error:', e);
            loadNext(index + 1);
          }
        }
        loadNext(0);
      }
    });
  <\/script>`;

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  ${POLICY_TAG}
  ${bootstrapScript}
  <style>${baseStyles}${listStyles}${viewStyles}${prismTheme}</style>
  ${katexStyles}
  ${prismScripts}
</head>
<body>${bodyContent}${listScript}${viewScripts}</body>
</html>`;
}

export { LIST_SEPARATOR };
