/**
 * The tiles and captions drawn on Tiled Captioned KSampler and Tiled Captioned Custom Sampler.
 *
 * While the node runs, the tiles being captioned or sampled are lit over its step preview;
 * afterwards the numbered tiles, the run's figures and each caption.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createPictureBand } from "./interface/picture_band.js";
import { PREVIEW_STATE, executionId, fetchOutputPreview } from "./interface/preview.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.TiledCaptionedSamplerUI";
const NODE_IDS = new Set(["WASTiledCaptionedKSampler", "WASTiledCaptionedCustomSampler"]);

// The message the node announces its active tiles with.
const ACTIVITY_EVENT = "was-tile-activity";

// The message ComfyUI sends a sampler's step preview with.
const PREVIEW_EVENT = "b_preview_with_metadata";

// Height of the panel in node units: the summary, the figures, the tile plan and the captions.
const PANEL_HEIGHT = 420;

// Height of the tile plan in CSS pixels when the panel opens.
const PLAN_HEIGHT = 220;

// Narrowest the panel is worth drawing in, in node units.
const MIN_WIDTH = 360;

// The widest figure name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 72;

// Outline of every tile, and the fill and edge of a tile being worked on.
const TILE_EDGE = "rgba(214,170,72,0.55)";
const ACTIVE_FILL = "rgba(255,196,64,0.30)";
const ACTIVE_EDGE = "rgba(255,214,96,1)";

const EMPTY_LABEL = "run the node once to see its tiles and their captions";

const UI_WIDGET_NAME = "was_tiled_captioned_sampler_ui";
const UI_WIDGET_TYPE = "was_tiled_captioned_sampler";

/**
 * Words naming what a run is doing to its active tiles.
 *
 * @param {object} live - The latest announcement.
 * @returns {string} A short line.
 */
function activityLabel(live) {
  const names = live.tiles.map((tile) => tile + 1).join(", ");
  if (live.phase === "caption") return `captioning tile ${names}`;
  if (live.phase === "sample" && live.steps) return `step ${live.step + 1} of ${live.steps}: tile ${names}`;
  return "";
}

/**
 * The share of a phase done, from 0 to 1.
 *
 * @param {object} live - The latest announcement.
 * @returns {number} The share, 0 where the phase has no count.
 */
function progressOf(live) {
  const count = live.boxes.length;
  if (live.phase === "caption" && count) return (Math.max(...live.tiles, -1) + 1) / count;
  if (live.phase === "sample" && live.steps) return live.step / live.steps;
  if (live.phase === "done") return 1;
  return 0;
}

/**
 * A report standing in for the node's own while it runs.
 *
 * @param {object} live - The latest announcement.
 * @returns {object} A report the panel draws.
 */
function liveReport(live) {
  const count = live.boxes.length;
  const done = Math.max(...live.tiles, -1) + 1;
  const lines = {
    prepare: "upscaling and encoding the picture",
    overview: "describing the whole picture",
    caption: `captioning tile ${done} of ${count}`,
    sample: `sampling step ${live.step + 1} of ${live.steps}`,
    done: "decoding the picture",
  };
  const counts = count ? [{ name: "tiles", value: count }] : [];
  if (live.phase === "caption") counts.push({ name: "captioned", value: `${done - 1} of ${count}` });
  if (live.phase === "sample") counts.push({ name: "step", value: `${live.step + 1} of ${live.steps}` });
  return { status: "ok", summary: lines[live.phase] || "running", counts };
}

/**
 * Draw every tile's outline over the picture, the active ones lit.
 *
 * @param {CanvasRenderingContext2D} pen - Where to draw.
 * @param {{x: number, y: number, width: number, height: number, ratio: number}} rect - The
 *   picture's drawn box.
 * @param {object} live - The latest announcement.
 * @returns {void}
 */
function drawTiles(pen, rect, live) {
  const box = ([top, bottom, left, right]) => [
    rect.x + left * rect.width,
    rect.y + top * rect.height,
    (right - left) * rect.width,
    (bottom - top) * rect.height,
  ];
  pen.save();
  pen.lineWidth = Math.max(1, rect.ratio);
  pen.strokeStyle = TILE_EDGE;
  for (const tile of live.boxes) pen.strokeRect(...box(tile));
  pen.lineWidth = Math.max(2, 2 * rect.ratio);
  for (const index of live.tiles) {
    const tile = live.boxes[index];
    if (!tile) continue;
    const [x, y, w, h] = box(tile);
    pen.fillStyle = ACTIVE_FILL;
    pen.fillRect(x, y, w, h);
    pen.strokeStyle = ACTIVE_EDGE;
    pen.strokeRect(x, y, w, h);
  }
  // How far through the phase the run is, along the picture's foot.
  const bar = Math.max(3, 4 * rect.ratio);
  pen.fillStyle = "rgba(14,16,22,0.75)";
  pen.fillRect(rect.x, rect.y + rect.height - bar, rect.width, bar);
  pen.fillStyle = ACTIVE_EDGE;
  pen.fillRect(rect.x, rect.y + rect.height - bar, rect.width * progressOf(live), bar);
  pen.restore();
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData?.name)) return;

    const proto = nodeType.prototype;
    if (proto.__was_tiled_captioned_sampler_wrapped) return;
    proto.__was_tiled_captioned_sampler_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        const live = { on: false, prompt: "", phase: "", tiles: [], boxes: [], step: 0, steps: 0 };
        let band = null;
        let panel = null;
        let shown = null;
        let shownUrl = "";

        const showLive = (image, url = "") => {
          if (!band || !image) return;
          if (shownUrl && shownUrl !== url) URL.revokeObjectURL(shownUrl);
          shown = image;
          shownUrl = url;
          band.show(image, activityLabel(live));
        };

        const onActivity = async (event) => {
          const fields = event?.detail;
          if (!fields || String(fields.node_id) !== String(executionId(node))) return;
          const fresh = String(fields.prompt_id ?? "") !== live.prompt;
          live.prompt = String(fields.prompt_id ?? "");
          live.phase = String(fields.phase ?? "");
          live.tiles = Array.isArray(fields.tiles) ? fields.tiles.map(Number) : [];
          live.boxes = Array.isArray(fields.boxes) ? fields.boxes : [];
          live.step = Number(fields.step) || 0;
          live.steps = Number(fields.steps) || 0;
          live.on = live.phase !== "done";
          if (fresh) shown = null;
          if (!shown && live.boxes.length) {
            // The tile plan is published as soon as the tiles are planned.
            const answer = await fetchOutputPreview(node, "", 0, live.prompt);
            if (answer.state === PREVIEW_STATE.READY && answer.image && !shown) showLive(answer.image);
          } else if (shown) {
            band?.show(shown, activityLabel(live));
          }
          panel?.update(liveReport(live));
          node.setDirtyCanvas?.(true, false);
        };

        const onPreview = async (event) => {
          const fields = event?.detail;
          if (!live.on || !fields?.blob) return;
          if (String(fields.nodeId ?? "") !== String(executionId(node))) return;
          const url = URL.createObjectURL(fields.blob);
          const image = new Image();
          image.src = url;
          try {
            await image.decode();
          } catch (error) {
            URL.revokeObjectURL(url);
            return;
          }
          if (!live.on) {
            URL.revokeObjectURL(url);
            return;
          }
          showLive(image, url);
        };

        api.addEventListener(ACTIVITY_EVENT, onActivity);
        api.addEventListener(PREVIEW_EVENT, onPreview);

        panel = createReportPanel(this, {
          className: "was-tiled-captioned-sampler",
          height: PANEL_HEIGHT,
          minWidth: MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the tile plan:",
          sketch: () => {
            band = createPictureBand(this, {
              label: "the tiles",
              height: PLAN_HEIGHT,
              overlay: (pen, rect) => {
                if (live.on && live.boxes.length) drawTiles(pen, rect, live);
              },
            });
            return band;
          },
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            api.removeEventListener(ACTIVITY_EVENT, onActivity);
            api.removeEventListener(PREVIEW_EVENT, onPreview);
            if (shownUrl) URL.revokeObjectURL(shownUrl);
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the tile plan:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the tile plan:`, error);
      }
      return result;
    };
  },
});
