/**
 * Base View Interface for ComfyUI Viewer
 * All view extensions should extend or follow this interface pattern.
 */

import { escapeHtml } from "../utils/helpers.mjs";

export { escapeHtml };

export class BaseView {
  /**
   * Unique identifier for this view type
   * @type {string}
   */
  static id = "base";

  /**
   * Display name for this view type
   * @type {string}
   */
  static displayName = "Base";

  /**
   * Priority for content type detection (higher = checked first)
   * @type {number}
   */
  static priority = 0;

  /**
   * Whether this view is a UI (interactive application).
   * UI views hide Edit/Clear/Download buttons, only showing Fullscreen.
   * @type {boolean}
   */
  static isUI = false;

  /**
   * Scripts loaded state
   * @type {object}
   */
  static scripts = {};

  /**
   * Check if this view can handle the given content
   * @param {string} content - The content to check
   * @returns {number} - Score indicating how well this view can handle the content (0 = cannot handle)
   */
  static detect(content) {
    return 0;
  }

  /**
   * Load any required scripts for this view
   * @param {string} basePath - Base path for loading scripts
   * @returns {Promise<boolean>} - True if scripts loaded successfully
   */
  static async loadScripts(basePath) {
    return true;
  }

  /**
   * Render the content to HTML
   * @param {string} content - The content to render
   * @param {object} theme - Theme tokens { bg, fg, border, accent }
   * @returns {string} - Rendered HTML body content
   */
  static render(content, theme) {
    return content;
  }

  /**
   * Get additional styles required for this view
   * @param {object} theme - Theme tokens
   * @returns {string} - CSS styles
   */
  static getStyles(theme) {
    return "";
  }

  /**
   * Get scripts to inject into the iframe
   * @returns {string} - Script tags/content
   */
  static getScripts() {
    return "";
  }

  /**
   * Check if scripts are ready
   * @returns {boolean}
   */
  static isReady() {
    return true;
  }

  /**
   * Get message types this view handles from iframe postMessage
   * @returns {string[]} - Array of message type strings
   */
  static getMessageTypes() {
    return [];
  }

  /**
   * Handle a message from the iframe
   * @param {string} messageType - The message type
   * @param {object} data - The message data
   * @param {object} node - The LiteGraph node
   * @param {object} app - The ComfyUI app instance
   * @returns {boolean} - True if message was handled
   */
  static handleMessage(messageType, data, node, app) {
    return false;
  }

  /**
   * Get view-specific state from node widgets
   * @param {object} node - The LiteGraph node
   * @returns {object|null} - State object or null
   */
  static getStateFromWidget(node) {
    return null;
  }

  /**
   * Inject saved state into content before rendering
   * @param {string} content - Original content
   * @param {object} state - State from getStateFromWidget
   * @returns {string} - Modified content with state injected
   */
  static injectState(content, state) {
    return content;
  }

  /**
   * Content marker prefix for this view type (if any)
   * @returns {string|null}
   */
  static getContentMarker() {
    return null;
  }

  /**
   * Whether this view uses the base iframe styles (iframe-base.css)
   * Views that provide their own complete styling should return false
   * @returns {boolean}
   */
  static usesBaseStyles() {
    return true;
  }
}

/**
 * Helper function to load a script file as text
 * @param {string} url - URL to fetch
 * @returns {Promise<string>}
 */
export async function loadScriptText(url) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.text() : "";
  } catch (e) {
    console.error(`[WAS Viewer] Failed to load script: ${url}`, e);
    return "";
  }
}

/**
 * Custom syntax highlighter using regex rules
 * Rules are applied in order; earlier rules take precedence
 * @param {string} content - The content to highlight
 * @param {Array<{name: string, pattern: RegExp, className: string}>} rules - Highlighting rules
 * @returns {string} - HTML with syntax highlighting spans
 */
export function highlightSyntax(content, rules) {
  const tokens = [];
  const escaped = escapeHtml(content);
  
  // Find all matches for all rules
  for (const rule of rules) {
    const regex = new RegExp(rule.pattern.source, rule.pattern.flags.includes('g') ? rule.pattern.flags : rule.pattern.flags + 'g');
    let match;
    while ((match = regex.exec(content)) !== null) {
      tokens.push({
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        className: rule.className,
        priority: rules.indexOf(rule)
      });
    }
  }
  
  // Sort by start position, then by priority (lower = higher priority)
  tokens.sort((a, b) => a.start - b.start || a.priority - b.priority);
  
  // Remove overlapping tokens (keep higher priority)
  const filtered = [];
  let lastEnd = 0;
  for (const token of tokens) {
    if (token.start >= lastEnd) {
      filtered.push(token);
      lastEnd = token.end;
    }
  }
  
  // Build highlighted output
  let result = '';
  let pos = 0;
  for (const token of filtered) {
    if (token.start > pos) {
      result += escapeHtml(content.slice(pos, token.start));
    }
    result += `<span class="${token.className}">${escapeHtml(token.text)}</span>`;
    pos = token.end;
  }
  if (pos < content.length) {
    result += escapeHtml(content.slice(pos));
  }
  
  return result;
}
