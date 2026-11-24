/**
 * A loop's live state, drawn on its End node.
 *
 * The counts and carried slots are refiled once per iteration under the original End node's id.
 */

import { createReportPanel } from "./report_panel.mjs";

const LOG_NAME = "WASNodeSuite.LoopState";

// Height in node units. Enough for the summary, the counts and eight slots without scrolling.
const PANEL_HEIGHT = 168;

/**
 * Build the panel one End node draws its state in.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @returns {{element: HTMLElement, height: number, refresh: () => void, dispose: () => void}}
 *   The panel, for `appendInterfaceWidget`.
 */
export function createLoopStatePanel(node) {
  return createReportPanel(node, {
    className: "was-loop-state",
    layout: "flow",
    tiles: false,
    height: PANEL_HEIGHT,
    logName: LOG_NAME,
    failure: "Failed to read the loop's state:",
  });
}
