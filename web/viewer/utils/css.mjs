/**
 * CSS Utilities - Load and inject CSS with theme variable support
 */

const loadedStyles = new Map();

/**
 * Load CSS file content
 * @param {string} url - URL to CSS file
 * @returns {Promise<string>}
 */
export async function loadCssFile(url) {
  if (loadedStyles.has(url)) {
    return loadedStyles.get(url);
  }
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load ${url}`);
    const css = await res.text();
    loadedStyles.set(url, css);
    return css;
  } catch (e) {
    console.error(`[WAS Viewer] Failed to load CSS: ${url}`, e);
    return "";
  }
}

/**
 * Generate CSS custom properties from theme object
 * @param {object} theme - Theme tokens { bg, fg, border, accent }
 * @returns {string} CSS custom properties
 */
export function themeToCssVars(theme) {
  return `
    :root {
      --theme-bg: ${theme.bg};
      --theme-fg: ${theme.fg};
      --theme-border: ${theme.border};
      --theme-accent: ${theme.accent};
    }
  `;
}

/**
 * Build CSS string with theme variables prepended
 * @param {string} css - CSS content
 * @param {object} theme - Theme tokens
 * @returns {string}
 */
export function buildThemedCss(css, theme) {
  return themeToCssVars(theme) + css;
}

/**
 * Load multiple CSS files and combine with theme
 * @param {string[]} urls - Array of CSS file URLs
 * @param {object} theme - Theme tokens
 * @returns {Promise<string>}
 */
export async function loadAndBuildCss(urls, theme) {
  const cssContents = await Promise.all(urls.map(url => loadCssFile(url)));
  const combinedCss = cssContents.join("\n");
  return buildThemedCss(combinedCss, theme);
}

/**
 * Get base path for CSS files
 * @returns {string}
 */
export function getCssBasePath() {
  const currentPath = import.meta.url;
  return currentPath.substring(0, currentPath.lastIndexOf("/utils/")) + "/css";
}
