/**
 * What a batching node made, drawn on the node itself.
 *
 * The panel shows the frame count, the frame size, the channel mode and the batch's memory cost.
 */

import { createReportPanel } from "./report_panel.mjs";

const LOG_NAME = "WASNodeSuite.BatchState";

// Height in node units the panel opens at, which shows every row without scrolling. The node's
// spare room is taken on top of it, so dragging the node taller gives the rows more air.
const PANEL_HEIGHT = 116;

/**
 * Build the panel one batching node draws its report in.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @returns {{element: HTMLElement, height: number, refresh: () => void, dispose: () => void}}
 *   The panel, for `appendInterfaceWidget`.
 */
export function createBatchStatePanel(node) {
  return createReportPanel(node, {
    className: "was-batch-state",
    height: PANEL_HEIGHT,
    logName: LOG_NAME,
    failure: "Failed to read the batch report:",
  });
}
