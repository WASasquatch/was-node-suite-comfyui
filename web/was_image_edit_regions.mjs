/**
 * The picture Image Edit Regions found things in, each find picked by a click and each edit
 * written on a card: what to do, the prompt, the paste and the shadow. Edits are kept in the
 * `edits` widget as JSON.
 */

import { app } from "../../scripts/app.js";
import { findWidget, writeWidget } from "./interface/audio_view.mjs";
import { PALETTE } from "./interface/box_layout.mjs";
import { drawLabel } from "./interface/canvas_label.mjs";
import { CARD_LOOK, CHIP_TONES, cardChip, cardRule } from "./interface/card.mjs";
import { createFloatingWindow } from "./interface/floating_window.mjs";
import { captureWheel, elementPoint, wheelPixels } from "./interface/pointer.mjs";
import {
  LABELS, PREVIEW_STATE, clientId, connected, executionId, fetchInputPreview, workflowId,
} from "./interface/preview.mjs";
import { fetchWithin } from "./interface/request.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onNodeFinished } from "./interface/run_events.mjs";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget, chainWidgetCallback, graphLoading, keepOnNode } from "./interface/widget.mjs";
import { SANS, el } from "./interface/window_kit.mjs";

const EXT_NAME = "WASNodeSuite.ImageEditRegionsUI";
const NODE_ID = "WASImageEditRegions";

const UI_WIDGET_NAME = "was_edit_regions_ui";
const UI_WIDGET_TYPE = "was_edit_regions";

const ROUTE = "/was/interface/api/edit_regions";
const EDITS_WIDGET = "edits";
const MODEL_WIDGET = "edit_model";

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 620;
const PANEL_WIDTH = 520;

// Panel width in CSS pixels from which the card sits beside the picture rather than under it.
const SIDE_BY_SIDE = 640;

// Width of the card beside the picture, and the least height of the picture above the card.
const CARD_WIDTH = 270;
const PICTURE_LEAST = 160;

// Node property holding whether every find is outlined.
const OUTLINES_KEY = "was_edit_regions_outlines";

// Milliseconds a prompt rests before it is written.
const SETTLE = 500;

// Most the picture is magnified, and how much one wheel notch changes it.
const MOST_ZOOM = 16;
const ZOOM_STEP = 1.0015;

// Finds the list shows before a search narrows it, and the most it shows after.
const LIST_UNSEARCHED = 24;
const LIST_MOST = 80;

// localStorage key the editor window keeps its place under.
const WINDOW_KEY = "was_edit_regions_window";

// A field, a section of the side column, and a chosen toggle, in the card's colours.
const FIELD = `box-sizing:border-box;min-width:0;padding:5px 8px;border-radius:6px;`
  + `border:1px solid ${CARD_LOOK.edge};background:${CARD_LOOK.inset};color:${CARD_LOOK.text};`
  + `font:12px/1.45 ${SANS};outline:none`;
const SECTION = `display:flex;flex-direction:column;gap:7px;padding:8px 9px 9px;border-radius:8px;`
  + `background:${CARD_LOOK.surface};border:1px solid ${CARD_LOOK.edge};box-shadow:${CARD_LOOK.shine};flex:0 0 auto`;
const CHOSEN = CHIP_TONES.accent;

/**
 * A section of the side column: a title over the card's rule, and a body.
 *
 * @param {string} title - The section's name.
 * @param {HTMLElement[]} [aside] - Chips or a swatch beside the title.
 * @returns {{element: HTMLElement, body: HTMLElement}} The section, and where its content goes.
 */
function section(title, aside = []) {
  const element = el("div", SECTION);
  const head = el("div", "display:flex;align-items:center;gap:6px;min-width:0");
  head.append(el("span", `font:600 10.5px/1.4 ${SANS};letter-spacing:0.06em;text-transform:uppercase;`
    + `color:${CARD_LOOK.title};white-space:nowrap`, title), ...aside);
  const body = el("div", "display:flex;flex-direction:column;gap:7px;min-width:0");
  element.append(head, cardRule(), body);
  return { element, body };
}

/**
 * A small rounded button in the card's colours.
 *
 * @param {string} label - Its words.
 * @param {() => void} onPress - Called on a press.
 * @param {{hint?: string, tone?: string, on?: boolean}} [options] - Hover text, `danger` for a
 *   removal, and whether it reads as switched on.
 * @returns {HTMLButtonElement} The button.
 */
function smallButton(label, onPress, { hint = "", tone = "", on = false } = {}) {
  const press = document.createElement("button");
  press.type = "button";
  press.textContent = label;
  if (hint) press.title = hint;
  const colour = tone === "danger" ? themeVar("error") : on ? CHOSEN.color : CARD_LOOK.muted;
  const rest = `flex:none;padding:2px 10px;border-radius:999px;cursor:pointer;white-space:nowrap;`
    + `font:600 11px/18px ${SANS};border:1px solid ${on ? CARD_LOOK.rule : CARD_LOOK.edge};`
    + `background:${on ? CHOSEN.background : CARD_LOOK.inset};color:${colour}`;
  press.style.cssText = rest;
  press.onmouseenter = () => {
    press.style.borderColor = tone === "danger" ? themeVar("error") : CARD_LOOK.rule;
    if (!on && tone !== "danger") press.style.color = CARD_LOOK.title;
  };
  press.onmouseleave = () => { press.style.cssText = rest; };
  press.addEventListener("click", (event) => {
    event.preventDefault();
    try {
      onPress();
    } catch (error) {
      console.error(`[${EXT_NAME}] ${label} failed:`, error);
    }
  });
  return press;
}

/**
 * A figure in a summary line, set apart from its words.
 *
 * @param {HTMLElement} line - The summary.
 * @param {number} value - The figure.
 * @param {string} words - What it counts.
 * @returns {void}
 */
function figure(line, value, words) {
  line.append(el("span", `font-weight:650;color:${CARD_LOOK.title}`, String(value)), words);
}

// Pixels a press may move and still be a click.
const CLICK_SLOP = 4;

// The words prompts name the picture by, per edit model.
const REFERENCES = { "MiniMax H3 Image": "<Picture 1>", other: "the picture" };

// What an edit does, the word on its button, and the label of its second field.
const ACTIONS = [
  ["edit", "Edit", ""],
  ["recolour", "Recolour", "to"],
  ["replace", "Replace", "with"],
  ["remove", "Remove", ""],
];

// Whether each action pastes through the box by default.
const PASTES_BOX = { edit: false, recolour: false, replace: true, remove: true };

/**
 * The prompt an action writes for a thing.
 *
 * @param {string} action - One of `ACTIONS`.
 * @param {string} thing - What is changed, as `red car`.
 * @param {string} ref - The words naming the picture.
 * @param {string} to - The colour or replacement.
 * @returns {string} The prompt.
 */
function template(action, thing, ref, to) {
  const what = thing || "area";
  const target = to.trim() || "...";
  if (action === "recolour") {
    return `Recolour the ${what} in ${ref} to ${target}, keeping its shape, material and folds.`;
  }
  if (action === "replace") {
    return `Replace the ${what} in ${ref} with ${target}, in the same place and at the same size.`;
  }
  if (action === "remove") {
    return `Remove the ${what} from ${ref} and fill in what was behind it so it matches its surroundings.`;
  }
  return `Change the ${what} in ${ref} to `;
}

/**
 * Shared area over joined area of two boxes held as shares of the picture.
 *
 * @param {number[]} first - `[left, top, right, bottom]`.
 * @param {number[]} second - The same.
 * @returns {number} 0 to 1.
 */
function boxOverlap(first, second) {
  const width = Math.max(0, Math.min(first[2], second[2]) - Math.max(first[0], second[0]));
  const height = Math.max(0, Math.min(first[3], second[3]) - Math.max(first[1], second[1]));
  const shared = width * height;
  const joined = (first[2] - first[0]) * (first[3] - first[1])
    + (second[2] - second[0]) * (second[3] - second[1]) - shared;
  return joined > 0 ? shared / joined : 0;
}

/**
 * The find a pick names, as the node matches it: its name with the closest box, else any find
 * with a box close enough.
 *
 * @param {{label: string, box: number[]}} pick - The pick.
 * @param {object[]} segments - The finds.
 * @returns {object|null} The find.
 */
function matchPick(pick, segments) {
  let best = null;
  let bestOverlap = 0;
  for (const segment of segments) {
    const overlap = boxOverlap(pick.box, segment.box);
    const named = !pick.label || segment.names.includes(pick.label);
    if (((named && overlap >= 0.5) || overlap >= 0.75) && overlap > bestOverlap) {
      best = segment;
      bestOverlap = overlap;
    }
  }
  return best;
}

/**
 * A base64 PNG of label layers decoded into one id plane per channel.
 *
 * @param {string} data - The PNG in base64.
 * @param {number} width - Its width.
 * @param {number} height - Its height.
 * @param {number} planes - How many of its three channels hold a layer.
 * @returns {Promise<Uint8Array[]>} The planes.
 */
async function decodeLayers(data, width, height, planes) {
  const image = new Image();
  image.src = `data:image/png;base64,${data}`;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const out = [];
  for (let channel = 0; channel < planes; channel += 1) {
    const plane = new Uint8Array(width * height);
    for (let at = 0; at < plane.length; at += 1) plane[at] = pixels[at * 4 + channel];
    out.push(plane);
  }
  return out;
}

/**
 * Build the editor, on the node or in its window.
 *
 * @param {object} node - The Image Edit Regions node.
 * @param {{inWindow?: boolean, onExpand?: () => void}} [options] - Whether it fills a window, and
 *   what the Expand button opens.
 * @returns {object} The panel `appendInterfaceWidget` hosts.
 */
function createEditor(node, options = {}) {
  let disposed = false;
  let finds = null;
  let layers = [];
  let byId = new Map();
  let picture = null;
  let pictureState = PREVIEW_STATE.WAITING;
  let outlines = null;
  let hover = null;
  let selected = -1;
  let cycle = null;
  let press = null;
  let lastText = null;
  let lastJobs = [];
  let unreadable = false;
  let settle = 0;
  let writing = false;
  let frame = 0;
  let serial = 0;
  let zoom = 1;
  let centre = [0.5, 0.5];
  let search = "";
  const tinted = new Map();

  const root = el("div", `display:flex;flex-direction:column;gap:7px;width:100%;height:100%;`
    + `box-sizing:border-box;padding:8px;font:12px ${SANS};color:${CARD_LOOK.text};outline:none;min-height:0`);
  root.tabIndex = 0;
  const bar = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;flex-wrap:wrap;min-width:0");
  const body = el("div", "display:flex;gap:8px;flex:1 1 auto;min-height:0;min-width:0");
  const stage = el("div", `position:relative;flex:1 1 auto;min-width:0;min-height:0;border-radius:7px;`
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};overflow:hidden`);
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;cursor:crosshair");
  stage.append(canvas);
  const card = el("div", `display:flex;flex-direction:column;gap:8px;flex:0 0 auto;min-height:0;min-width:0;`
    + `overflow:auto;padding:1px;scrollbar-width:thin;scrollbar-color:${themeVar("scrollbarThumb")} transparent`);
  body.append(stage, card);
  root.append(bar, cardRule(), body);

  const ctx = canvas.getContext("2d");

  /**
   * The edits the widget holds, read again only when its text changed.
   *
   * @returns {object[]} The jobs.
   */
  function readJobs() {
    const text = String(findWidget(node, EDITS_WIDGET)?.value ?? "");
    if (text === lastText) return lastJobs;
    lastText = text;
    unreadable = false;
    selected = -1;
    cycle = null;
    if (!text.trim()) {
      lastJobs = [];
      return lastJobs;
    }
    try {
      const data = JSON.parse(text);
      lastJobs = Array.isArray(data?.jobs) ? data.jobs.filter((job) => job && typeof job === "object") : [];
    } catch (error) {
      lastJobs = [];
      unreadable = true;
    }
    return lastJobs;
  }

  /**
   * Write the edits as one undoable change.
   *
   * @param {object[]} jobs - Every job.
   * @param {number} [pick] - The job to leave selected.
   * @param {boolean} [keepCard] - Leave the card as drawn, for a write from one of its own fields.
   * @returns {void}
   */
  function commit(jobs, pick = selected, keepCard = false) {
    if (unreadable) return;
    const text = jobs.length ? JSON.stringify({ jobs }) : "";
    lastText = text;
    lastJobs = jobs;
    selected = Math.min(pick, jobs.length - 1);
    writing = true;
    try {
      writeWidget(node, EDITS_WIDGET, text);
    } finally {
      writing = false;
    }
    if (!keepCard) drawCard();
    drawBar();
    redraw();
  }

  /**
   * The words prompts name the picture by, from the node's edit model.
   *
   * @returns {string} The words.
   */
  function reference() {
    return REFERENCES[String(findWidget(node, MODEL_WIDGET)?.value ?? "")] ?? REFERENCES.other;
  }

  /**
   * The finds under a point, smallest first.
   *
   * @param {number} u - Across, as a share of the picture.
   * @param {number} v - Down, as a share of the picture.
   * @returns {object[]} The finds.
   */
  function findsAt(u, v) {
    if (!finds || u < 0 || v < 0 || u >= 1 || v >= 1) return [];
    const [width, height] = finds.map;
    const at = Math.floor(v * height) * width + Math.floor(u * width);
    const found = [];
    for (const plane of layers) {
      const id = plane[at];
      if (id && byId.has(id)) found.push(byId.get(id));
    }
    return found.sort((first, second) => first.area - second.area);
  }

  /**
   * Where the picture is drawn in the canvas, in CSS pixels.
   *
   * @returns {{x: number, y: number, width: number, height: number}|null} The rectangle.
   */
  function view() {
    const width = finds?.width ?? picture?.naturalWidth;
    const height = finds?.height ?? picture?.naturalHeight;
    const boxWidth = canvas.clientWidth;
    const boxHeight = canvas.clientHeight;
    if (!width || !height || boxWidth < 2 || boxHeight < 2) return null;
    const scale = Math.min(boxWidth / width, boxHeight / height) * zoom;
    const drawnWidth = width * scale;
    const drawnHeight = height * scale;
    return {
      x: boxWidth / 2 - centre[0] * drawnWidth,
      y: boxHeight / 2 - centre[1] * drawnHeight,
      width: drawnWidth,
      height: drawnHeight,
    };
  }

  /**
   * Hold the centre where the magnified picture still covers the view.
   *
   * @returns {void}
   */
  function clampCentre() {
    const half = 0.5 / zoom;
    centre = [Math.min(1 - half, Math.max(half, centre[0])), Math.min(1 - half, Math.max(half, centre[1]))];
  }

  /**
   * Magnify the picture about a point, keeping that point under the pointer.
   *
   * @param {number} factor - The change in magnification.
   * @param {{u: number, v: number, x: number, y: number}} point - Where the pointer is.
   * @returns {boolean} Whether the magnification moved.
   */
  function zoomAt(factor, point) {
    const next = Math.min(MOST_ZOOM, Math.max(1, zoom * factor));
    if (next === zoom || !view()) return false;
    zoom = next;
    const drawn = view();
    centre = [
      point.u - (point.x - canvas.clientWidth / 2) / drawn.width,
      point.v - (point.y - canvas.clientHeight / 2) / drawn.height,
    ];
    clampCentre();
    drawBar();
    redraw();
    return true;
  }

  /**
   * One find's mask at the layers' size in one colour, kept for reuse.
   *
   * @param {number} id - The find.
   * @param {string} colour - A CSS colour.
   * @returns {HTMLCanvasElement|null} The mask.
   */
  function maskOf(id, colour) {
    const key = `${id}|${colour}`;
    if (tinted.has(key)) return tinted.get(key);
    if (!finds) return null;
    const [width, height] = finds.map;
    const shape = document.createElement("canvas");
    shape.width = width;
    shape.height = height;
    const shapeCtx = shape.getContext("2d");
    const pixels = shapeCtx.createImageData(width, height);
    for (const plane of layers) {
      for (let at = 0; at < plane.length; at += 1) {
        if (plane[at] === id) pixels.data[at * 4 + 3] = 255;
      }
    }
    shapeCtx.putImageData(pixels, 0, 0);
    shapeCtx.globalCompositeOperation = "source-in";
    shapeCtx.fillStyle = colour;
    shapeCtx.fillRect(0, 0, width, height);
    tinted.set(key, shape);
    return shape;
  }

  /**
   * Every find's edge, drawn once per run.
   *
   * @returns {void}
   */
  function buildOutlines() {
    outlines = null;
    if (!finds || !layers.length) return;
    const [width, height] = finds.map;
    const top = new Uint16Array(width * height);
    for (let at = 0; at < top.length; at += 1) {
      let best = 0;
      let bestArea = Infinity;
      for (const plane of layers) {
        const id = plane[at];
        const area = id ? byId.get(id)?.area ?? Infinity : Infinity;
        if (id && area < bestArea) {
          best = id;
          bestArea = area;
        }
      }
      top[at] = best;
    }
    const shape = document.createElement("canvas");
    shape.width = width;
    shape.height = height;
    const shapeCtx = shape.getContext("2d");
    const pixels = shapeCtx.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const at = y * width + x;
        const id = top[at];
        const edge = (x + 1 < width && top[at + 1] !== id) || (y + 1 < height && top[at + width] !== id);
        if (edge && (id || top[at + 1] || top[at + width])) {
          pixels.data[at * 4] = 255;
          pixels.data[at * 4 + 1] = 255;
          pixels.data[at * 4 + 2] = 255;
          pixels.data[at * 4 + 3] = 150;
        }
      }
    }
    shapeCtx.putImageData(pixels, 0, 0);
    outlines = shape;
  }

  /**
   * The finds a job names on this run, and its colour.
   *
   * @param {object} job - The job.
   * @param {number} index - Its place.
   * @returns {{ids: number[], colour: string}} Them.
   */
  function jobFinds(job, index) {
    const segments = finds?.segments ?? [];
    const ids = [];
    for (const pick of Array.isArray(job.picks) ? job.picks : []) {
      const found = matchPick(pick, segments);
      if (found && !ids.includes(found.id)) ids.push(found.id);
    }
    return { ids, colour: PALETTE[index % PALETTE.length] };
  }

  /**
   * Repaint on the next frame.
   *
   * @returns {void}
   */
  function redraw() {
    if (frame || disposed) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      paint();
    });
  }

  /**
   * Paint the picture, the edits, the hovered find and the outlines.
   *
   * @returns {void}
   */
  function paint() {
    const ratio = surfaceRatio(canvas);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    const theme = readTheme();
    const place = view();
    const jobs = readJobs();
    if (!place || !picture) {
      ctx.fillStyle = theme.fgMuted;
      ctx.font = `12px ${SANS}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(emptyWords(), canvas.clientWidth / 2, canvas.clientHeight / 2);
      return;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(picture, place.x, place.y, place.width, place.height);
    if (outlines && wantsOutlines()) ctx.drawImage(outlines, place.x, place.y, place.width, place.height);
    jobs.forEach((job, index) => {
      const { ids, colour } = jobFinds(job, index);
      ctx.globalAlpha = index === selected ? 0.5 : 0.26;
      for (const id of ids) {
        const shape = maskOf(id, colour);
        if (shape) ctx.drawImage(shape, place.x, place.y, place.width, place.height);
      }
      ctx.globalAlpha = 1;
      for (const point of Array.isArray(job.points) ? job.points : []) {
        ctx.beginPath();
        ctx.arc(place.x + point[0] * place.width, place.y + point[1] * place.height, 5, 0, Math.PI * 2);
        ctx.fillStyle = colour;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = theme.panelBg;
        ctx.stroke();
      }
      const anchor = ids.length ? byId.get(ids[0])?.box : null;
      const spot = anchor
        ? [place.x + anchor[0] * place.width + 4, place.y + anchor[1] * place.height + 10]
        : Array.isArray(job.points?.[0])
          ? [place.x + job.points[0][0] * place.width + 8, place.y + job.points[0][1] * place.height]
          : null;
      if (spot) {
        drawLabel(ctx, { width: canvas.clientWidth, height: canvas.clientHeight, top: 0, theme },
          `${index + 1} ${job.label || ""}`.trim(), spot[0], spot[1], { swatch: colour });
      }
    });
    if (hover?.finds?.length) {
      const target = hover.finds[hover.depth % hover.finds.length];
      const shape = maskOf(target.id, theme.accent);
      if (shape) {
        ctx.globalAlpha = 0.38;
        ctx.drawImage(shape, place.x, place.y, place.width, place.height);
        ctx.globalAlpha = 1;
      }
      const next = hover.finds.length > 1 ? hover.finds[(hover.depth + 1) % hover.finds.length] : null;
      const picked = readJobs().some((job, index) => jobFinds(job, index).ids.includes(target.id));
      const hint = picked ? " · right-click removes" : next ? ` · click again: ${next.label}` : "";
      const words = `${describe(target)}${hint}`;
      drawLabel(ctx, { width: canvas.clientWidth, height: canvas.clientHeight, top: 0, theme },
        words, hover.x + 12, hover.y + 14, { swatch: theme.accent });
    }
  }

  /**
   * What the picture area says while it has no picture.
   *
   * @returns {string} The words.
   */
  function emptyWords() {
    if (!connected()) return LABELS[PREVIEW_STATE.CONNECTING];
    if (pictureState === PREVIEW_STATE.LOADING) return LABELS[PREVIEW_STATE.LOADING];
    if (pictureState === PREVIEW_STATE.FAILED) return LABELS[PREVIEW_STATE.FAILED];
    return "Run the graph to find the things in the picture";
  }

  /**
   * A find in words: its colour and name.
   *
   * @param {object} find - The find.
   * @returns {string} As `red car`.
   */
  function describe(find) {
    return [find.colour, find.label].filter(Boolean).join(" ");
  }

  /**
   * Whether every find is outlined.
   *
   * @returns {boolean} The node's choice, on by default.
   */
  function wantsOutlines() {
    return node.properties?.[OUTLINES_KEY] !== false;
  }

  /**
   * The header: the title, what the run found, how many edits there are, and the view buttons.
   *
   * @returns {void}
   */
  function drawBar() {
    bar.replaceChildren();
    const jobs = readJobs();
    const title = el("span", `flex:none;font:600 12.5px/20px ${SANS};color:${CARD_LOOK.title}`, "Edit regions");
    const summary = el("span", `flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`
      + `font:12px/20px ${SANS};color:${CARD_LOOK.muted}`);
    if (finds) figure(summary, finds.segments.length, " things · ");
    else summary.append("Not run yet · ");
    figure(summary, jobs.length, jobs.length === 1 ? " edit" : " edits");
    bar.append(title, summary);
    if (unreadable) bar.append(cardChip("edits unreadable", "warning"));
    const notes = finds?.notes ?? [];
    if (notes.length) {
      const warn = cardChip(`${notes.length} passed over`, "warning");
      warn.title = notes.join("\n");
      bar.append(warn);
    }
    bar.append(el("span", "flex:1 1 auto"));
    if (zoom > 1) {
      bar.append(smallButton(`${zoom.toFixed(1)}x · Fit`, () => resetZoom(), { hint: "Show the whole picture", on: true }));
    }
    bar.append(smallButton("Outlines", () => {
      keepOnNode(node, OUTLINES_KEY, !wantsOutlines());
      drawBar();
      redraw();
    }, { hint: "Outline every find on the picture", on: wantsOutlines() }));
    if (options.onExpand) {
      bar.append(smallButton("Expand", () => options.onExpand(), { hint: "Open the editor in a window" }));
    }
  }

  /**
   * Show the whole picture again.
   *
   * @returns {void}
   */
  function resetZoom() {
    zoom = 1;
    centre = [0.5, 0.5];
    drawBar();
    redraw();
  }

  /**
   * The side column: the selected edit, the list of edits and the finds.
   *
   * @returns {void}
   */
  function drawCard() {
    card.replaceChildren();
    const jobs = readJobs();
    const job = jobs[selected];
    if (job) {
      const swatch = el("span", `flex:none;width:9px;height:9px;border-radius:3px;background:${PALETTE[selected % PALETTE.length]}`);
      const part = section(`Edit ${selected + 1}`, [swatch]);
      jobEditor(part.body, job);
      card.append(part.element);
    } else {
      const part = section("Pick a thing");
      part.body.append(el("div", `color:${CARD_LOOK.muted};line-height:1.5`,
        unreadable
          ? "The edits text could not be read. It is kept as it is; clear the edits widget to start again."
          : "Click a thing to edit it, click again for what holds it, Shift-click to add to the selected "
            + "edit, right-click to remove. The wheel zooms and a drag pans."));
      card.append(part.element);
    }
    if (jobs.length) {
      const part = section("Edits", [cardChip(String(jobs.length), "accent")]);
      jobList(part.body, jobs);
      card.append(part.element);
    }
    if (finds?.segments?.length) {
      const part = section("Find", [cardChip(String(finds.segments.length), "plain")]);
      findList(part.body, jobs);
      card.append(part.element);
    }
  }

  /**
   * The finds by name, narrowed by a search, each chip picking its find as a click on it does.
   *
   * @param {HTMLElement} host - Where the search and chips go.
   * @param {object[]} jobs - Every job.
   * @returns {void}
   */
  function findList(host, jobs) {
    const field = el("input", FIELD);
    field.placeholder = "Search, as cup";
    field.value = search;
    const chips = el("div", "display:flex;flex-wrap:wrap;gap:4px;min-width:0");
    const fill = () => {
      chips.replaceChildren();
      const words = search.trim().toLowerCase();
      const owned = new Set(jobs.flatMap((job, index) => jobFinds(job, index).ids));
      const matching = finds.segments.filter((find) => !words
        || find.names.some((name) => name.includes(words)) || (find.colour ?? "").includes(words));
      const shown = matching.slice(0, words ? LIST_MOST : LIST_UNSEARCHED);
      for (const find of shown) {
        const mine = owned.has(find.id);
        const rest = `flex:none;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`
          + `padding:1px 9px;border-radius:999px;cursor:pointer;font:11.5px/19px ${SANS};`
          + `border:1px solid ${mine ? CARD_LOOK.rule : CARD_LOOK.edge};`
          + `background:${mine ? CHOSEN.background : CARD_LOOK.inset};color:${mine ? CHOSEN.color : CARD_LOOK.text}`;
        const chip = el("span", rest, describe(find));
        chip.title = `${find.names.slice(0, 4).join(", ")} · ${Math.round(find.score * 100)}%`;
        chip.onmouseenter = () => {
          chip.style.borderColor = CARD_LOOK.rule;
          showFind(find);
        };
        chip.onmouseleave = () => {
          chip.style.cssText = rest;
          hover = null;
          redraw();
        };
        chip.addEventListener("click", (event) => pickFind(find, event.shiftKey));
        chips.append(chip);
      }
      if (matching.length > shown.length) {
        chips.append(el("span", `flex:none;padding:1px 4px;font:11.5px/19px ${SANS};color:${CARD_LOOK.muted}`,
          `${matching.length - shown.length} more`));
      }
    };
    field.addEventListener("input", () => {
      search = field.value;
      fill();
    });
    fill();
    host.append(field, chips);
  }

  /**
   * Light one find from the list and bring it into view when the picture is magnified.
   *
   * @param {object} find - The find.
   * @returns {void}
   */
  function showFind(find) {
    const place = view();
    if (!place) return;
    if (zoom > 1) {
      centre = [(find.box[0] + find.box[2]) / 2, (find.box[1] + find.box[3]) / 2];
      clampCentre();
    }
    const drawn = view();
    hover = {
      finds: [find],
      depth: 0,
      x: drawn.x + find.box[2] * drawn.width,
      y: drawn.y + find.box[1] * drawn.height,
    };
    redraw();
  }

  /**
   * The fields of the selected edit.
   *
   * @param {HTMLElement} host - Where they go.
   * @param {object} job - The selected edit.
   * @returns {void}
   */
  function jobEditor(host, job) {
    const thing = el("input", `${FIELD};font-weight:600`);
    thing.value = String(job.label ?? "");
    thing.title = "Named in the prompt";
    host.append(thing);

    const actions = el("div", `display:flex;flex-wrap:wrap;gap:3px;padding:3px;border-radius:999px;`
      + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well}`);
    for (const [action, word] of ACTIONS) {
      const chosen = (job.action ?? "edit") === action;
      const press = el("button", `flex:1 1 auto;padding:3px 10px;border-radius:999px;border:none;cursor:pointer;`
        + `font:600 11px/17px ${SANS};background:${chosen ? CHOSEN.background : "transparent"};`
        + `color:${chosen ? CHOSEN.color : CARD_LOOK.muted}`, word);
      press.type = "button";
      press.addEventListener("click", () => update({ action, paste: undefined }));
      actions.append(press);
    }
    host.append(actions);

    const second = ACTIONS.find(([action]) => action === (job.action ?? "edit"))?.[2] ?? "";
    let toField = null;
    if (second) {
      const row = el("div", "display:flex;align-items:center;gap:7px;min-width:0");
      toField = el("input", `${FIELD};flex:1 1 auto`);
      toField.value = String(job.to ?? "");
      toField.placeholder = job.action === "recolour" ? "mint green" : "a ginger cat";
      row.append(el("span", `flex:none;color:${CARD_LOOK.muted}`, second), toField);
      host.append(row);
    }

    const prompt = el("textarea", `${FIELD};width:100%;resize:vertical;line-height:1.5`);
    prompt.rows = 4;
    prompt.spellcheck = false;
    prompt.value = String(job.prompt ?? "");
    host.append(prompt);
    if (job.custom) {
      const again = el("div", "display:flex");
      again.append(smallButton("Write it from the action", () => update({ custom: false })));
      host.append(again);
    }

    const parts = el("div", "display:flex;gap:4px;flex-wrap:wrap;align-items:center;min-width:0");
    parts.append(el("span", `flex:none;font:600 10.5px/1.4 ${SANS};letter-spacing:0.06em;text-transform:uppercase;`
      + `color:${CARD_LOOK.muted}`, "Parts"));
    const segments = finds?.segments ?? [];
    (Array.isArray(job.picks) ? job.picks : []).forEach((pick, at) => {
      const found = matchPick(pick, segments);
      parts.append(partChip(found ? describe(found) : pick.label || "thing", () => {
        dropOrUpdate({ picks: job.picks.filter((_, other) => other !== at) });
      }));
    });
    (Array.isArray(job.points) ? job.points : []).forEach((_, at) => {
      parts.append(partChip(`point ${at + 1}`, () => {
        dropOrUpdate({ points: job.points.filter((__, other) => other !== at) });
      }));
    });
    host.append(parts);

    const options = el("div", "display:flex;align-items:center;gap:12px;flex-wrap:wrap");
    const toggle = (words, checked, onChange, hint = "") => {
      const label = el("label", `display:flex;align-items:center;gap:5px;cursor:pointer;color:${CARD_LOOK.text}`);
      const box = el("input", `margin:0;accent-color:${themeVar("cardAccent")}`);
      box.type = "checkbox";
      box.checked = checked;
      box.addEventListener("change", () => onChange(box.checked));
      if (hint) label.title = hint;
      label.append(box, el("span", "", words));
      return label;
    };
    options.append(
      toggle("Include its shadow", Boolean(job.shadow), (on) => update({ shadow: on })),
      toggle("Paste its box", pastesBox(job), (on) => update({ paste: on ? "box" : "shape" }), "Pastes the whole box"),
    );
    host.append(options);

    const foot = el("div", "display:flex;gap:6px;align-items:center");
    foot.append(smallButton("Delete edit", () => removeJob(selected), { tone: "danger" }));
    host.append(foot);

    /**
     * Change the edit's parts, or take the edit away once it has none.
     *
     * @param {object} change - The new picks or points.
     * @returns {void}
     */
    function dropOrUpdate(change) {
      const next = { ...job, ...change };
      if (!(next.picks?.length) && !(next.points?.length)) {
        removeJob(selected);
        return;
      }
      update(change);
    }

    /**
     * Change the selected edit and write it, its prompt rewritten unless it was typed.
     *
     * @param {object} change - The fields to change.
     * @param {boolean} [typed] - The change came from the prompt box, which keeps its focus.
     * @returns {void}
     */
    function update(change, typed = false) {
      const current = readJobs();
      if (!current[selected]) return;
      const next = { ...current[selected], ...change };
      if (change.paste === undefined && "paste" in change) delete next.paste;
      if (!next.custom) next.prompt = template(next.action ?? "edit", next.label ?? "", reference(), String(next.to ?? ""));
      const all = current.slice();
      all[selected] = next;
      commit(all, selected, typed);
    }

    thing.addEventListener("change", () => update({ label: thing.value.trim(), labelTyped: true }));
    toField?.addEventListener("change", () => update({ to: toField.value }));
    prompt.addEventListener("input", () => {
      clearTimeout(settle);
      settle = setTimeout(() => update({ prompt: prompt.value, custom: true }, true), SETTLE);
    });
    prompt.addEventListener("change", () => {
      clearTimeout(settle);
      if (prompt.value !== String(readJobs()[selected]?.prompt ?? "")) update({ prompt: prompt.value, custom: true }, true);
    });
  }

  /**
   * A part of an edit as a chip, with a cross that removes it.
   *
   * @param {string} text - What the part is.
   * @param {() => void} onRemove - Called by the cross.
   * @returns {HTMLElement} The chip.
   */
  function partChip(text, onRemove) {
    const chip = el("span", `display:inline-flex;align-items:center;gap:1px;padding:0 3px 0 9px;max-width:100%;`
      + `border-radius:999px;border:1px solid ${CARD_LOOK.edge};background:${CARD_LOOK.inset};`
      + `font:11.5px/19px ${SANS};color:${CARD_LOOK.text}`);
    chip.append(el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap", text));
    const cross = el("button", `flex:none;width:17px;height:17px;padding:0;border:none;border-radius:999px;cursor:pointer;`
      + `background:transparent;color:${CARD_LOOK.muted};font:13px/17px ${SANS}`, "×");
    cross.type = "button";
    cross.title = `Remove ${text}`;
    cross.onmouseenter = () => { cross.style.color = themeVar("error"); };
    cross.onmouseleave = () => { cross.style.color = CARD_LOOK.muted; };
    cross.addEventListener("click", (event) => {
      event.preventDefault();
      onRemove();
    });
    chip.append(cross);
    return chip;
  }

  /**
   * Take the thing or the point under the pointer out of the edit holding it.
   *
   * @param {{u: number, v: number, x: number, y: number}} point - Where the pointer is.
   * @returns {void}
   */
  function unpickAt(point) {
    if (unreadable) return;
    const jobs = readJobs().slice();
    const place = view();
    let owner = -1;
    let change = null;
    jobs.forEach((job, index) => {
      if (change || !place) return;
      const points = Array.isArray(job.points) ? job.points : [];
      const near = points.findIndex(([u, v]) => Math.hypot(place.x + u * place.width - point.x, place.y + v * place.height - point.y) <= 8);
      if (near >= 0) {
        owner = index;
        change = { points: points.filter((_, at) => at !== near) };
      }
    });
    if (!change) {
      for (const find of findsAt(point.u, point.v)) {
        owner = jobs.findIndex((job, index) => jobFinds(job, index).ids.includes(find.id));
        if (owner < 0) continue;
        const picks = jobs[owner].picks.filter((pick) => !matchPick(pick, [find]));
        change = { picks };
        break;
      }
    }
    if (!change) return;
    const next = { ...jobs[owner], ...change };
    if (!(next.picks?.length) && !(next.points?.length)) {
      jobs.splice(owner, 1);
      commit(jobs, selected === owner ? -1 : selected > owner ? selected - 1 : selected);
      return;
    }
    jobs[owner] = next;
    commit(jobs, owner);
  }

  /**
   * Whether an edit is pasted through its box.
   *
   * @param {object} job - The job.
   * @returns {boolean} Its own choice, else its action's.
   */
  function pastesBox(job) {
    if (job.paste === "box" || job.paste === "shape") return job.paste === "box";
    return PASTES_BOX[job.action ?? "edit"] ?? false;
  }

  /**
   * Every edit as a row that selects it.
   *
   * @param {HTMLElement} host - Where the rows go.
   * @param {object[]} jobs - Every job.
   * @returns {void}
   */
  function jobList(host, jobs) {
    jobs.forEach((job, index) => {
      const chosen = index === selected;
      const rest = `display:flex;align-items:center;gap:7px;padding:4px 7px;border-radius:6px;cursor:pointer;min-width:0;`
        + `background:${chosen ? CHOSEN.background : "transparent"};color:${chosen ? CHOSEN.color : CARD_LOOK.text}`;
      const row = el("div", rest);
      row.append(el("span", `flex:none;width:8px;height:8px;border-radius:3px;background:${PALETTE[index % PALETTE.length]}`));
      const verb = ACTIONS.find(([action]) => action === (job.action ?? "edit"))?.[1] ?? "Edit";
      const target = job.to ? ` ${job.action === "replace" ? "with" : "to"} ${job.to}` : "";
      const words = el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0");
      words.append(el("span", `font-weight:600;color:${chosen ? CHOSEN.color : CARD_LOOK.title}`, `${index + 1}. ${verb}`),
        ` ${job.label || "area"}${target}`);
      row.append(words);
      row.title = String(job.prompt ?? "");
      row.onmouseenter = () => { if (!chosen) row.style.background = CARD_LOOK.inset; };
      row.onmouseleave = () => { row.style.cssText = rest; };
      row.addEventListener("click", () => {
        selected = index;
        drawCard();
        redraw();
      });
      host.append(row);
    });
  }

  /**
   * Take one edit away.
   *
   * @param {number} index - Its place.
   * @returns {void}
   */
  function removeJob(index) {
    const jobs = readJobs().slice();
    if (index < 0 || index >= jobs.length) return;
    jobs.splice(index, 1);
    commit(jobs, -1);
  }

  /**
   * A pick naming a find, as the node stores it.
   *
   * @param {object} find - The find.
   * @returns {{label: string, box: number[]}} The pick.
   */
  function pickOf(find) {
    return { label: find.label, box: find.box.slice() };
  }

  /**
   * Answer a click on the picture.
   *
   * @param {number} u - Across, as a share of the picture.
   * @param {number} v - Down, as a share of the picture.
   * @param {boolean} adding - Shift was held.
   * @returns {void}
   */
  function clickAt(u, v, adding) {
    if (unreadable || u < 0 || v < 0 || u > 1 || v > 1) return;
    const jobs = readJobs().slice();
    const under = findsAt(u, v);
    const again = cycle && Math.abs(cycle.u - u) < 0.004 && Math.abs(cycle.v - v) < 0.004;
    if (adding && jobs[selected]) {
      const job = { ...jobs[selected] };
      if (under.length) {
        const find = under[0];
        const picks = Array.isArray(job.picks) ? job.picks.slice() : [];
        const held = picks.findIndex((pick) => matchPick(pick, [find]));
        if (held >= 0) picks.splice(held, 1);
        else picks.push(pickOf(find));
        job.picks = picks;
      } else {
        job.points = [...(Array.isArray(job.points) ? job.points : []), [round(u), round(v)]];
      }
      jobs[selected] = job;
      commit(jobs);
      return;
    }
    if (again && under.length > 1 && jobs[selected]?.picks?.length === 1 && !jobs[selected]?.points?.length) {
      cycle.depth = (cycle.depth + 1) % under.length;
      const find = under[cycle.depth];
      const job = { ...jobs[selected], picks: [pickOf(find)] };
      if (!job.labelTyped) job.label = describe(find);
      if (!job.custom) job.prompt = template(job.action ?? "edit", job.label, reference(), String(job.to ?? ""));
      jobs[selected] = job;
      if (hover) hover.depth = cycle.depth;
      commit(jobs);
      return;
    }
    if (under.length) {
      const owner = jobs.findIndex((job, index) => jobFinds(job, index).ids.includes(under[0].id));
      if (owner >= 0) {
        selected = owner;
        cycle = { u, v, depth: 0 };
        drawCard();
        redraw();
        return;
      }
    }
    const find = under[0];
    const label = find ? describe(find) : "area";
    const job = {
      picks: find ? [pickOf(find)] : [],
      points: find ? [] : [[round(u), round(v)]],
      action: "edit",
      label,
      prompt: template("edit", label, reference(), ""),
    };
    jobs.push(job);
    cycle = { u, v, depth: 0 };
    commit(jobs, jobs.length - 1);
  }

  /**
   * Pick a find from the list, as a click on it on the picture picks it.
   *
   * @param {object} find - The find.
   * @param {boolean} adding - Shift was held.
   * @returns {void}
   */
  function pickFind(find, adding) {
    if (unreadable) return;
    const jobs = readJobs().slice();
    if (adding && jobs[selected]) {
      const job = { ...jobs[selected] };
      const picks = Array.isArray(job.picks) ? job.picks.slice() : [];
      const held = picks.findIndex((pick) => matchPick(pick, [find]));
      if (held >= 0) picks.splice(held, 1);
      else picks.push(pickOf(find));
      job.picks = picks;
      jobs[selected] = job;
      commit(jobs);
      return;
    }
    const owner = jobs.findIndex((job, index) => jobFinds(job, index).ids.includes(find.id));
    if (owner >= 0) {
      selected = owner;
      drawCard();
      redraw();
      return;
    }
    const label = describe(find);
    jobs.push({
      picks: [pickOf(find)], points: [], action: "edit", label, prompt: template("edit", label, reference(), ""),
    });
    commit(jobs, jobs.length - 1);
  }

  /**
   * A share kept to five places.
   *
   * @param {number} value - The share.
   * @returns {number} Rounded.
   */
  function round(value) {
    return Math.round(value * 1e5) / 1e5;
  }

  /**
   * The share of the picture under a pointer event.
   *
   * @param {PointerEvent} event - The event.
   * @returns {{u: number, v: number, x: number, y: number}|null} Where it is.
   */
  function pointOf(event) {
    const place = view();
    if (!place) return null;
    const point = elementPoint(canvas, event);
    if (!point) return null;
    return { u: (point.x - place.x) / place.width, v: (point.y - place.y) / place.height, x: point.x, y: point.y };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button === 2) {
      event.preventDefault();
      event.stopPropagation();
      const point = pointOf(event);
      if (point) unpickAt(point);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    root.focus({ preventScroll: true });
    const at = elementPoint(canvas, event);
    press = { x: event.clientX, y: event.clientY, at, centre: centre.slice(), shift: event.shiftKey, panning: false };
    canvas.setPointerCapture?.(event.pointerId);
  });
  const endPress = (event, cancelled) => {
    const held = press;
    press = null;
    try {
      canvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the pointer:`, error);
    }
    if (!held || cancelled || held.panning) return;
    if (Math.hypot(event.clientX - held.x, event.clientY - held.y) > CLICK_SLOP) return;
    const point = pointOf(event);
    if (point) clickAt(point.u, point.v, held.shift);
  };
  canvas.addEventListener("pointerup", (event) => endPress(event, false));
  canvas.addEventListener("pointercancel", (event) => endPress(event, true));
  canvas.addEventListener("lostpointercapture", () => { press = null; });
  canvas.addEventListener("pointermove", (event) => {
    if (press && event.buttons === 0) press = null;
    if (press && zoom > 1 && (press.panning || Math.hypot(event.clientX - press.x, event.clientY - press.y) > CLICK_SLOP)) {
      const place = view();
      const at = elementPoint(canvas, event);
      if (place) {
        press.panning = true;
        centre = [
          press.centre[0] - (at.x - press.at.x) / place.width,
          press.centre[1] - (at.y - press.at.y) / place.height,
        ];
        clampCentre();
        hover = null;
        canvas.style.cursor = "grabbing";
        redraw();
      }
      return;
    }
    canvas.style.cursor = "crosshair";
    const point = pointOf(event);
    const under = point ? findsAt(point.u, point.v) : [];
    const depth = cycle && point && Math.abs(cycle.u - point.u) < 0.004 && Math.abs(cycle.v - point.v) < 0.004
      ? cycle.depth : 0;
    hover = under.length ? { finds: under, depth, x: point.x, y: point.y } : null;
    redraw();
  });
  canvas.addEventListener("pointerleave", () => {
    hover = null;
    redraw();
  });
  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    event.stopPropagation();
    resetZoom();
  });
  root.addEventListener("contextmenu", (event) => event.preventDefault());
  root.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(event.target?.tagName ?? "");
    if (typing) return;
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      event.stopPropagation();
      if (selected >= 0) removeJob(selected);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      selected = -1;
      cycle = null;
      drawCard();
      redraw();
    }
  });

  const releaseWheel = captureWheel(root, (event) => {
    if (event.target === canvas) {
      const point = pointOf(event);
      if (!point || !finds) return false;
      return zoomAt(ZOOM_STEP ** -wheelPixels(event, canvas).y, point);
    }
    if (!card.contains(event.target) || card.scrollHeight <= card.clientHeight) return false;
    const before = card.scrollTop;
    card.scrollTop += wheelPixels(event, card).y;
    return card.scrollTop !== before;
  });

  /**
   * Lay the card beside the picture on a wide panel and under it on a narrow one.
   *
   * @returns {void}
   */
  function arrange() {
    const wide = root.clientWidth >= SIDE_BY_SIDE;
    body.style.flexDirection = wide ? "row" : "column";
    card.style.width = wide ? `${CARD_WIDTH}px` : "auto";
    card.style.maxHeight = wide ? "none" : "45%";
    stage.style.minHeight = wide ? "0" : `${Math.min(PICTURE_LEAST, root.clientHeight / 2)}px`;
    redraw();
  }

  const resizeWatch = new ResizeObserver(() => arrange());
  resizeWatch.observe(root);
  const stopRatio = watchSurfaceRatio(canvas, () => redraw());
  const stopTheme = onThemeChange(() => {
    tinted.clear();
    redraw();
  });

  /**
   * Read the run's finds and the picture again.
   *
   * @returns {Promise<void>} Settles once both are read.
   */
  async function load() {
    const ticket = ++serial;
    const id = executionId(node);
    if (!id) return;
    pictureState = PREVIEW_STATE.LOADING;
    redraw();
    try {
      const query = `${ROUTE}?node_id=${encodeURIComponent(id)}`
        + `&workflow=${encodeURIComponent(workflowId() ?? "")}`
        + `&client_id=${encodeURIComponent(clientId())}&node_type=${encodeURIComponent(NODE_ID)}`;
      const [answer, shown] = await Promise.all([fetchWithin(query), fetchInputPreview(node)]);
      if (ticket !== serial || disposed) return;
      let payload = null;
      if (answer.status === 200) payload = await answer.json();
      let planes = [];
      if (payload) {
        const [width, height] = payload.map;
        let remaining = payload.layer_count;
        for (const data of payload.layers) {
          const count = Math.min(3, remaining);
          planes = planes.concat(await decodeLayers(data, width, height, count));
          remaining -= count;
        }
      }
      if (ticket !== serial || disposed) return;
      finds = payload;
      layers = planes;
      byId = new Map((payload?.segments ?? []).map((segment) => [segment.id, segment]));
      tinted.clear();
      buildOutlines();
      picture = shown?.state === PREVIEW_STATE.READY ? shown.image : null;
      pictureState = shown?.state ?? PREVIEW_STATE.FAILED;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the finds:`, error);
      pictureState = PREVIEW_STATE.FAILED;
    }
    drawBar();
    drawCard();
    redraw();
  }

  const stopFinished = onNodeFinished(node, () => load());
  chainWidgetCallback(node, EDITS_WIDGET, () => {
    if (writing) return;
    drawBar();
    drawCard();
    redraw();
  }, EXT_NAME);

  drawBar();
  drawCard();
  if (options.inWindow) load();

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: () => load(),
    dispose() {
      disposed = true;
      serial += 1;
      clearTimeout(settle);
      if (frame) cancelAnimationFrame(frame);
      resizeWatch.disconnect();
      stopRatio?.();
      stopTheme?.();
      stopFinished?.();
      releaseWheel?.();
    },
  };
}

// The window open on each node.
const windows = new Map();

/**
 * Open the editor in a window beside the graph, or bring the open one forward.
 *
 * @param {object} node - The node.
 * @returns {void}
 */
function openWindow(node) {
  const open = windows.get(node);
  if (open) {
    open.window.open();
    return;
  }
  let entry = null;
  const close = () => {
    entry?.editor.dispose();
    entry?.window.dispose();
    windows.delete(node);
  };
  const win = createFloatingWindow({
    title: "Image Edit Regions",
    badge: `${node.title ?? "Image Edit Regions"} #${node.id}`,
    width: 1240,
    height: 820,
    minWidth: 560,
    minHeight: 400,
    storageKey: WINDOW_KEY,
    logName: EXT_NAME,
    onClose: close,
  });
  const editor = createEditor(node, { inWindow: true });
  editor.element.style.flex = "1 1 auto";
  editor.element.style.minHeight = "0";
  win.body.appendChild(editor.element);
  entry = { window: win, editor, close };
  windows.set(node, entry);
  win.open();
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_edit_regions_wrapped) return;
    proto.__was_edit_regions_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        const panel = createEditor(node, { onExpand: () => openWindow(node) });
        appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            if (!graphLoading()) windows.get(node)?.close();
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
