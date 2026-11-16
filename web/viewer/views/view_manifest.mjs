/**
 * View Manifest for ComfyUI Viewer
 *
 * This file lists all view modules to be loaded.
 * Extensions can add their own views by importing and calling registerView()
 * after views are loaded, or by creating their own manifest.
 *
 * To add a new view:
 * 1. Create a view file (e.g., myview.mjs) that exports a class extending BaseView
 * 2. Add the filename to VIEW_FILES array below
 */

export const VIEW_FILES = [
  "canvas.mjs",
  "html.mjs",
  "svg.mjs",
  "markdown.mjs",
  "json.mjs",
  "csv.mjs",
  "yaml.mjs",
  "ansi.mjs",
  "python.mjs",
  "javascript.mjs",
  "css.mjs",
  "object.mjs",
  "text.mjs",
];
