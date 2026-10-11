/**
 * Every node panel read again once a workflow has finished loading.
 *
 * Fires after a workflow switch, an undo, a redo or a page load has rebuilt the graph and
 * restored its run outputs.
 */

import { app } from "../../scripts/app.js";
import { beginGraphLoad, endGraphLoad } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.PanelReload";

// How long after the graph is configured a frontend with no `afterLoadGraph` hook is given to
// restore its outputs, in milliseconds.
const FALLBACK_MS = 400;

let fallback = 0;

app.registerExtension({
  name: EXT_NAME,

  async beforeLoadGraph() {
    clearTimeout(fallback);
    beginGraphLoad();
  },

  async afterConfigureGraph() {
    clearTimeout(fallback);
    fallback = setTimeout(() => {
      try {
        endGraphLoad();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to refresh the panels after a load:`, error);
      }
    }, FALLBACK_MS);
  },

  async afterLoadGraph() {
    clearTimeout(fallback);
    try {
      endGraphLoad();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to refresh the panels after a load:`, error);
    }
  },
});
