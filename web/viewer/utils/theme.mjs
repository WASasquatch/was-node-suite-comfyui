/**
 * Theme utilities for ComfyUI Viewer
 * Extracts theme tokens from ComfyUI CSS variables
 */

/**
 * Read a CSS variable value from a computed style
 * @param {CSSStyleDeclaration} style - Computed style object
 * @param {string} name - CSS variable name
 * @returns {string} Variable value or empty string
 */
export function readCssVar(style, name) {
  const v = style.getPropertyValue(name);
  return v ? String(v).trim() : "";
}

/**
 * Compute basic theme tokens from ComfyUI CSS variables
 * @returns {{bg: string, fg: string, border: string, accent: string}}
 */
export function computeThemeTokens() {
  const rootStyle = getComputedStyle(document.documentElement);
  const bodyStyle = getComputedStyle(document.body);

  const bg =
    readCssVar(rootStyle, "--comfy-menu-bg") ||
    readCssVar(rootStyle, "--background-color") ||
    readCssVar(rootStyle, "--bg-color") ||
    bodyStyle.backgroundColor ||
    "#1a1a1a";
  const fg =
    readCssVar(rootStyle, "--input-text") ||
    readCssVar(rootStyle, "--text-color") ||
    readCssVar(rootStyle, "--fg-color") ||
    bodyStyle.color ||
    "#e0e0e0";
  const border =
    readCssVar(rootStyle, "--border-color") ||
    readCssVar(rootStyle, "--comfy-border-color") ||
    "#444";
  const accent =
    readCssVar(rootStyle, "--primary-color") ||
    readCssVar(rootStyle, "--accent-color") ||
    readCssVar(rootStyle, "--comfy-accent") ||
    "#4a9eff";

  return { bg, fg, border, accent };
}

/**
 * Extract all ComfyUI theme CSS variables for iframe injection
 * Returns comprehensive theme object with all available colors
 * @returns {Object} Full theme object with all color tokens
 */
export function getFullTheme() {
  const rootStyle = getComputedStyle(document.documentElement);
  const bodyStyle = getComputedStyle(document.body);
  
  // Helper to get first available value
  const getVar = (...names) => {
    for (const name of names) {
      const val = readCssVar(rootStyle, name);
      if (val) return val;
    }
    return null;
  };
  
  return {
    // Core colors
    bg: getVar("--comfy-menu-bg", "--background-color", "--bg-color") || bodyStyle.backgroundColor || "#1a1a1a",
    bgLight: getVar("--comfy-input-bg", "--input-bg") || "#2a2a2a",
    bgDark: getVar("--comfy-menu-secondary-bg") || "#151515",
    
    fg: getVar("--input-text", "--text-color", "--fg-color") || bodyStyle.color || "#e0e0e0",
    fgMuted: getVar("--descrip-text", "--text-muted") || "#888888",
    fgDisabled: getVar("--disabled-text") || "#666666",
    
    border: getVar("--border-color", "--comfy-border-color") || "#444444",
    borderLight: getVar("--tr-odd-bg-color") || "#3a3a3a",
    
    accent: getVar("--p-button-text-primary-color", "--primary-color", "--accent-color") || "#4a9eff",
    accentHover: getVar("--p-button-text-primary-hover-color") || "#5ab0ff",
    accentBg: getVar("--p-highlight-background") || "#4a9eff22",
    
    success: getVar("--success-color") || "#4caf50",
    warning: getVar("--warning-color") || "#ff9800",
    error: getVar("--error-color", "--error-text") || "#f44336",
    
    // Input/Control colors
    inputBg: getVar("--comfy-input-bg", "--input-bg") || "#2a2a2a",
    inputBorder: getVar("--p-form-field-border-color") || "#555555",
    inputFocus: getVar("--p-form-field-focus-border-color") || "#4a9eff",
    
    // Panel/Surface colors
    panelBg: getVar("--comfy-menu-bg") || "#252525",
    panelHeader: getVar("--section-header-bg") || "#333333",
    
    // Shadow
    shadow: getVar("--shadow-color") || "rgba(0,0,0,0.3)",
    
    // Scrollbar
    scrollbarThumb: getVar("--comfy-scrollbar-thumb") || "#555555",
    scrollbarTrack: getVar("--comfy-scrollbar-track") || "#1a1a1a",
    
    // Selection
    selection: getVar("--p-highlight-background") || "#4a9eff44",
    selectionText: getVar("--p-highlight-color") || "#ffffff",
  };
}

/**
 * Generate CSS custom properties string from theme object
 * @param {Object} theme - Theme object from getFullTheme()
 * @returns {string} CSS custom properties declarations wrapped in :root
 */
export function themeToCssVars(theme) {
  const vars = Object.entries(theme)
    .map(([key, value]) => `--theme-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value};`)
    .join('\n      ');
  return `:root {\n      ${vars}\n    }\n    `;
}
