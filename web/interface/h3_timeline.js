/**
 * A MiniMax H3 run drawn as tracks under a time ruler: scenes, pinned frames, references.
 *
 * Scenes are laid out by `timeline` in `h3_extend.js`. A gesture reports once on release, a
 * dropped file where it lands.
 */

import {
  AUDIO_CARRY, AUDIO_REFERENCE, BRIDGING, CONTINUITY, CUT_LIKE, CUT_PICTURES, FPS, REFERENCE_VIDEO,
  bridgeFrames, closingIndex, cutsInto, durationOf, framesOf, guideLength, resolvedIndex,
  resolvedTransition, snapClip, snapOverlap, timeline,
} from "./h3_extend.js";
import { DRAG_TYPE, carriesFiles } from "./asset_browser.js";
import { drawSheetFrame } from "./segment_previews.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme } from "./theme.js";

const LOG_NAME = "WASNodeSuite.H3Timeline";

// The track header column, the space before time zero, and the room kept past the last scene.
const GUTTER = 118;
const LEAD = 14;
const TAIL = 150;

// The ruler, the gap between tracks, and the tracks in drawing order, in CSS pixels.
const RULER = 28;
const LANE_GAP = 6;
const LANES = Object.freeze([
  { id: "scenes", label: "Scenes", height: 64 },
  { id: "frames", label: "Pinned frames", height: 56 },
  { id: "refs", label: "References", height: 34 },
  { id: "shared", label: "All scenes", height: 32 },
]);

// The scroll bar along the bottom.
const BAR_HEIGHT = 8;

/** The height the tracks need, ruler and scroll bar included. */
export const TIMELINE_HEIGHT = RULER + 8
  + LANES.reduce((sum, lane) => sum + lane.height + LANE_GAP, 0) + BAR_HEIGHT + 8;

// A scene clip's corner radius, the transition button's radius, and the trim handle's grab.
const RADIUS = 7;
const SEAM_RADIUS = 11;
const GRAB = 6;

// A pinned frame's thumbnail, and a reference chip.
const PIN_WIDTH = 48;
const PIN_HEIGHT = 32;

// Least space between two pinned-frame labels, in CSS pixels.
const LABEL_GAP = 4;
const CHIP_HEIGHT = 24;
const CHIP_THUMB = 18;
const CHIP_MAX = 170;

// The add-scene button past the last clip.
const ADD_WIDTH = 104;

// Zoom bounds, in CSS pixels per second.
const MIN_ZOOM = 4;
const MAX_ZOOM = 600;

// Tick spacings a ruler chooses from, in seconds, and the most labels it draws.
const TICK_STEPS = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
const MAX_TICKS = 16;

// Movement before a press becomes a drag, and how near an edge a drop snaps to it.
const MOVE_THRESHOLD = 4;
const SNAP_PX = 12;

// The largest backing store the canvas is given.
const MAX_EDGE = 8192;

// Shortest and longest a scene may be dragged to, in seconds.
const MIN_SECONDS = 0.2;
const MAX_SECONDS = 150;

const SANS = "system-ui,sans-serif";
const MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";

/** How a scene joins the one before it, as each transition is named and drawn. */
export const TRANSITIONS = Object.freeze({
  carry: { glyph: "»", name: "Continue the shot", detail: "one unbroken shot" },
  refresh: { glyph: "↻", name: "Continue, refreshed", detail: "the same shot, carried frames re-noised" },
  handoff: { glyph: "⇥", name: "Cut on the last frame", detail: "a new shot opening on the last frame" },
  "reference (video)": { glyph: "▶", name: "Cut, reference last frames", detail: "a new shot referencing the last frames" },
  "reference (sample)": { glyph: "◎", name: "Cut, keep the cast", detail: "a new shot referencing stills from the whole clip" },
  cut: { glyph: "✂", name: "Hard cut", detail: "a new scene, nothing carried" },
  [AUDIO_CARRY]: { glyph: "♪", name: "Cut, keep the sound", detail: "new picture, the soundtrack carries on" },
  [AUDIO_REFERENCE]: {
    glyph: "♫", name: "Cut, keep the sound and cast",
    detail: "new picture referencing the last frames, the soundtrack carries on",
  },
});

/** The colour a scene takes from how it joins the one before, where no row colour is chosen. */
// Muted, apart from the row colours: blues continue the shot, violets and roses cut and keep the
// cast, sand cuts clean, ambers cut under the carried sound.
export const TRANSITION_TINTS = Object.freeze({
  opening: { label: "Opening scene", stripe: "#8e97a6", fill: "rgba(142, 151, 166, 0.30)", text: "#f3f5f8" },
  carry: { label: "Continue the shot", stripe: "#7b9cbc", fill: "rgba(123, 156, 188, 0.32)", text: "#eef4fa" },
  refresh: { label: "Continue, refreshed", stripe: "#7aa79a", fill: "rgba(122, 167, 154, 0.32)", text: "#eef8f5" },
  handoff: { label: "Cut on the last frame", stripe: "#9890bf", fill: "rgba(152, 144, 191, 0.32)", text: "#f3f1fb" },
  "reference (video)": {
    label: "Cut, reference last frames", stripe: "#ad89b3", fill: "rgba(173, 137, 179, 0.32)", text: "#f8f0f9",
  },
  "reference (sample)": {
    label: "Cut, keep the cast", stripe: "#b78a98", fill: "rgba(183, 138, 152, 0.32)", text: "#faf0f3",
  },
  cut: { label: "Hard cut", stripe: "#ab9c80", fill: "rgba(171, 156, 128, 0.30)", text: "#f8f5ee" },
  [AUDIO_CARRY]: {
    label: "Cut, keep the sound", stripe: "#bf9a62", fill: "rgba(191, 154, 98, 0.32)", text: "#fbf4e9",
  },
  [AUDIO_REFERENCE]: {
    label: "Cut, keep the sound and cast", stripe: "#b98466", fill: "rgba(185, 132, 102, 0.32)", text: "#faf0ea",
  },
});

/**
 * The transition colour a scene takes when it has no row colour of its own.
 *
 * @param {number} index - The scene, from 0.
 * @param {string} continuity - The row's continuity.
 * @param {number} overlap - The row's overlap in frames.
 * @param {string} [sound="auto"] - The row's sound choice.
 * @returns {{label: string, stripe: string, fill: string, text: string}} A `TRANSITION_TINTS` entry.
 */
export function sceneTint(index, continuity, overlap, sound = "auto") {
  return TRANSITION_TINTS[tintKey(index, continuity, overlap, sound)];
}

/**
 * The `TRANSITION_TINTS` key a scene's colour is read from.
 *
 * @param {number} index - The scene, from 0.
 * @param {string} continuity - The row's continuity.
 * @param {number} overlap - The row's overlap in frames.
 * @param {string} [sound="auto"] - The row's sound choice.
 * @returns {string} A key of `TRANSITION_TINTS`.
 */
export function tintKey(index, continuity, overlap, sound = "auto") {
  if (index <= 0) return "opening";
  const effective = effectiveTransition(continuity, overlap, sound);
  const key = effective;
  return Object.hasOwn(TRANSITION_TINTS, key) ? key : CONTINUITY[0];
}

/**
 * The last frame a scene shows before the next one takes over.
 *
 * @param {Array<object|null>} layout - Entries from `timeline`.
 * @param {number} index - The scene, from 0.
 * @returns {number} The frame after its last, in the finished clip.
 */
export function drawnEndOf(layout, index) {
  const entry = layout[index];
  const next = layout[index + 1];
  if (!next) return entry.end;
  return Math.min(entry.end, next.start + (next.trimmed > 0 ? 0 : next.head));
}

/**
 * The scene showing at a frame of the finished clip.
 *
 * @param {Array<object|null>} layout - Entries from `timeline`.
 * @param {number} frame - A frame of the finished clip.
 * @returns {number} The scene, from 0, or -1 past the last.
 */
export function sceneAt(layout, frame) {
  for (let index = 0; index < layout.length; index += 1) {
    if (!layout[index]) break;
    if (frame >= layout[index].start && frame < drawnEndOf(layout, index)) return index;
  }
  return -1;
}

/**
 * The frame of a scene's sheet that lands on a frame of the finished clip.
 *
 * @param {object} entry - The scene's entry from `timeline`.
 * @param {object} fields - The sheet's fields.
 * @param {number} frame - A frame of the finished clip.
 * @returns {number} A frame of the sheet; the ones before the scene's new frames are skipped.
 */
export function sheetFrameAt(entry, fields, frame) {
  const offset = Math.max(0, (Number(fields.frames) || 0) - entry.newFrames);
  return offset + Math.max(0, Math.round(frame) - entry.start);
}

/** What each part an asset plays is called on screen. */
export const ROLE_NAMES = Object.freeze({
  "first frame": "Opening frame",
  "last frame": "Closing frame",
  keyframe: "Keyframe",
  "reference picture": "Picture reference",
  "reference clip": "Clip reference",
  "reference audio": "Sound reference",
});

// What a gesture is doing.
const GRIP = {
  PLAYHEAD: "playhead", END: "end", SEAM: "seam", HEAD: "head", CLIP: "clip", PIN: "pin",
  BAR: "bar",
};

/**
 * The continuity a row's setting comes to, with an overlap of 0 read as a cut.
 *
 * @param {string} continuity - The row's continuity.
 * @param {number} overlap - The row's overlap in frames.
 * @param {string} [sound="auto"] - The row's sound choice.
 * @returns {string} A key of `TRANSITIONS`.
 */
export function effectiveTransition(continuity, overlap, sound = "auto") {
  const named = Object.hasOwn(TRANSITIONS, continuity) ? String(continuity) : CONTINUITY[0];
  const { picture, bridged } = resolvedTransition(named, sound);
  if (bridged) return picture === "cut" ? AUDIO_CARRY : picture === REFERENCE_VIDEO ? AUDIO_REFERENCE : picture;
  if (!CUT_PICTURES.includes(picture) && overlap <= 0) return "cut";
  return picture;
}

/**
 * The kind of file a drag holds, from what the page reports during the drag.
 *
 * @param {DragEvent} event - The drag event.
 * @returns {string} `picture`, `clip`, `sound`, or an empty string when it cannot be told.
 */
function draggedKind(event) {
  for (const item of Array.from(event?.dataTransfer?.items ?? [])) {
    const type = String(item?.type ?? "");
    if (type.startsWith("image/")) return "picture";
    if (type.startsWith("video/")) return "clip";
    if (type.startsWith("audio/")) return "sound";
  }
  return "";
}

/**
 * The tick spacing for a span of seconds.
 *
 * @param {number} seconds - Seconds on screen.
 * @returns {number} One of `TICK_STEPS`.
 */
function tickStep(seconds) {
  const wanted = Math.max(seconds, 1e-6) / MAX_TICKS;
  return TICK_STEPS.find((step) => step >= wanted) ?? TICK_STEPS[TICK_STEPS.length - 1];
}

/**
 * A time written for the ruler.
 *
 * @param {number} seconds - The time.
 * @param {number} step - The tick spacing.
 * @returns {string} As `12s`, `0.5s` or `1:05`.
 */
function rulerLabel(seconds, step) {
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${String(Math.round(rest)).padStart(2, "0")}`;
  }
  const places = step >= 1 ? 0 : step >= 0.5 ? 1 : 2;
  return `${seconds.toFixed(places)}s`;
}

/**
 * A time written as a clock, for the playhead.
 *
 * @param {number} seconds - The time.
 * @returns {string} As `0:02.50`.
 */
export function clockOf(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

/**
 * Trace a rounded rectangle.
 *
 * @param {CanvasRenderingContext2D} pen - The context.
 * @param {number} x - Left edge.
 * @param {number} y - Top edge.
 * @param {number} w - Width.
 * @param {number} h - Height.
 * @param {number} r - Corner radius.
 * @returns {void}
 */
function trace(pen, x, y, w, h, r) {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  pen.beginPath();
  if (typeof pen.roundRect === "function") {
    pen.roundRect(x, y, w, h, radius);
    return;
  }
  pen.moveTo(x + radius, y);
  pen.arcTo(x + w, y, x + w, y + h, radius);
  pen.arcTo(x + w, y + h, x, y + h, radius);
  pen.arcTo(x, y + h, x, y, radius);
  pen.arcTo(x, y, x + w, y, radius);
  pen.closePath();
}

/**
 * A text shortened with an ellipsis to fit a width.
 *
 * @param {CanvasRenderingContext2D} pen - The context, its font set.
 * @param {string} text - The text.
 * @param {number} most - The widest it may be drawn, in CSS pixels.
 * @returns {string} The text, whole where it fits, else cut and ended with an ellipsis.
 */
function ellipsized(pen, text, most) {
  const whole = String(text ?? "");
  if (most <= 0) return "";
  if (pen.measureText(whole).width <= most) return whole;
  let low = 0;
  let high = whole.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (pen.measureText(`${whole.slice(0, middle)}…`).width <= most) low = middle;
    else high = middle - 1;
  }
  return low ? `${whole.slice(0, low).trimEnd()}…` : "";
}

/**
 * Draw a picture filling a box, cropped to its shape.
 *
 * @param {CanvasRenderingContext2D} pen - The context.
 * @param {HTMLImageElement} picture - The picture.
 * @param {number} x - Left edge.
 * @param {number} y - Top edge.
 * @param {number} w - Width.
 * @param {number} h - Height.
 * @returns {void}
 */
function drawCovered(pen, picture, x, y, w, h) {
  const scale = Math.max(w / picture.naturalWidth, h / picture.naturalHeight);
  const sw = w / scale;
  const sh = h / scale;
  const sx = (picture.naturalWidth - sw) / 2;
  const sy = (picture.naturalHeight - sh) / 2;
  pen.drawImage(picture, sx, sy, sw, sh, x, y, w, h);
}

/**
 * The glyph standing in for a file with no picture.
 *
 * @param {string} kind - `picture`, `clip` or `sound`.
 * @returns {string} The glyph.
 */
function kindGlyph(kind) {
  if (kind === "sound") return "♪";
  if (kind === "clip") return "▶";
  return "▣";
}

/**
 * Build the tracks.
 *
 * @param {object} options - Settings.
 * @param {() => object} options.read - Answers `{rows, assets, selected, selectedAsset,
 *   playhead, dragging}`: `rows` as `[{seconds, overlap, continuity, source, title, colour,
 *   model, warning}]`, `assets` as `[{id, segment, role, frame, name, kind, thumb, frames,
 *   tag}]`, `selected` a row index or null, `selectedAsset` an asset id or null, `playhead` in
 *   seconds, `dragging` the `{kind}` of a file being dragged from the media bin, or null.
 * @param {(index: number) => void} options.onSelect - A scene was pressed.
 * @param {(id: number|null) => void} options.onSelectAsset - An asset was pressed.
 * @param {(change: object) => void} options.onChange - A gesture finished, as
 *   `{type: "duration", index, seconds}`, `{type: "overlap", index, frames}`,
 *   `{type: "move", from, to}`, `{type: "keyframe", id, frame}`, `{type: "playhead", seconds}`,
 *   `{type: "add"}` or `{type: "edit", index}`.
 * @param {(target: object, event: DragEvent) => void} options.onDrop - A file was dropped, with
 *   where it lands: `{action, scene, segment, role, frame, label}`, `action` one of `pin`,
 *   `reference` and `new-scene`.
 * @param {(index: number, event: PointerEvent) => void} [options.onMenu] - A scene was right
 *   clicked.
 * @param {(index: number, event: PointerEvent) => void} [options.onTransitionMenu] - A
 *   transition button was pressed.
 * @param {(id: number, event: PointerEvent) => void} [options.onAssetMenu] - An asset was
 *   right clicked.
 * @param {string} [options.logName] - What a failure is logged under.
 * @returns {{element: HTMLElement, repaint: Function, fit: Function, zoomBy: Function,
 *   reveal: Function, dispose: Function}} The tracks.
 */
export function createTimeline(options = {}) {
  const read = typeof options.read === "function" ? options.read : () => null;
  const call = (name) => (typeof options[name] === "function" ? options[name] : () => {});
  const onSelect = call("onSelect");
  const onSelectAsset = call("onSelectAsset");
  const onChange = call("onChange");
  const onDrop = call("onDrop");
  const onMenu = call("onMenu");
  const onTransitionMenu = call("onTransitionMenu");
  const onAssetMenu = call("onAssetMenu");
  const logName = options.logName || LOG_NAME;

  const root = document.createElement("div");
  root.className = "was-h3-timeline";
  root.style.cssText = "position:relative;width:100%;height:100%;min-height:0;overflow:hidden;"
    + "touch-action:none;user-select:none;outline:none";
  root.tabIndex = 0;
  root.setAttribute("aria-label", "Timeline");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  root.appendChild(canvas);

  let disposed = false;
  let zoom = 40;
  let scroll = 0;
  let manual = false;
  let regions = [];
  let hover = null;
  let drag = null;
  let dropping = null;
  const thumbs = new Map();

  /**
   * The picture a thumbnail URL names, loaded once.
   *
   * @param {string} url - The picture's URL.
   * @returns {HTMLImageElement|null} The picture, or null until it has loaded.
   */
  const thumbnail = (url) => {
    if (!url) return null;
    let held = thumbs.get(url);
    if (held === undefined) {
      held = { image: new Image(), ready: false };
      held.image.onload = () => {
        held.ready = held.image.naturalWidth > 0;
        paint();
      };
      held.image.onerror = () => {
        held.ready = false;
      };
      held.image.src = url;
      thumbs.set(url, held);
    }
    return held.ready ? held.image : null;
  };

  /**
   * What the tracks draw, with every figure sane.
   *
   * @returns {object|null} The rows, their layout, the assets, the selection and the playhead.
   */
  const current = () => {
    let value = null;
    try {
      value = read();
    } catch (error) {
      console.error(`[${logName}] The timeline accessor failed:`, error);
      return null;
    }
    if (!value) return null;
    const rows = (value.rows ?? []).map((row) => ({
      seconds: Math.max(MIN_SECONDS, Number(row.seconds) || MIN_SECONDS),
      overlap: Math.max(0, Math.round(Number(row.overlap) || 0)),
      continuity: row.continuity || CONTINUITY[0],
      source: Number.isFinite(Number(row.source)) ? Number(row.source) : -1,
      title: String(row.title ?? ""),
      colour: row.colour ?? null,
      preview: row.preview ?? null,
      model: String(row.model ?? ""),
      sound: String(row.sound ?? "auto"),
      warning: String(row.warning ?? ""),
    }));
    return withLayout({
      rows,
      assets: Array.isArray(value.assets) ? value.assets : [],
      selected: Number.isInteger(value.selected) ? value.selected : null,
      selectedAsset: value.selectedAsset ?? null,
      playhead: Math.max(0, Number(value.playhead) || 0),
      dragging: value.dragging ?? null,
    });
  };

  /**
   * A state with its rows laid out.
   *
   * @param {object} state - Rows and the rest.
   * @returns {object} The state, with `layout` up to the first row the nodes refuse, and `total`.
   */
  const withLayout = (state) => {
    let layout = [];
    try {
      layout = state.rows.length ? timeline(state.rows) : [];
    } catch (error) {
      console.error(`[${logName}] The timeline could not be laid out:`, error);
    }
    const refused = layout.indexOf(null);
    const placed = refused < 0 ? layout : layout.slice(0, refused);
    return {
      ...state,
      layout: placed,
      refused: refused < 0 ? null : refused,
      total: placed.length ? placed[placed.length - 1].end : 0,
    };
  };

  const width = () => root.clientWidth || 1;
  const height = () => root.clientHeight || 1;

  /** Where a frame sits across the tracks, in CSS pixels. */
  const xOf = (frame) => GUTTER + LEAD + (frame / FPS - scroll) * zoom;

  /** The frame under a position across the tracks, fractional. */
  const frameAt = (x) => ((x - GUTTER - LEAD) / zoom + scroll) * FPS;

  /**
   * Where each track sits.
   *
   * @returns {object} Each lane's `{id, label, top, height}`, keyed by id.
   */
  const lanes = () => {
    const placed = {};
    let top = RULER + 8;
    for (const lane of LANES) {
      placed[lane.id] = { ...lane, top };
      top += lane.height + LANE_GAP;
    }
    return placed;
  };

  /**
   * The lane a height falls in.
   *
   * @param {number} y - The height, in CSS pixels.
   * @returns {string|null} The lane's id, or null above or below every lane.
   */
  const laneAt = (y) => {
    for (const lane of Object.values(lanes())) {
      if (y >= lane.top - LANE_GAP / 2 && y < lane.top + lane.height + LANE_GAP / 2) return lane.id;
    }
    return null;
  };

  /**
   * Fit the whole run into the tracks.
   *
   * @returns {void}
   */
  const fit = () => {
    manual = false;
    paint();
  };

  /**
   * Zoom so the whole run fills the tracks.
   *
   * @param {object} state - What `current` answered.
   * @returns {void}
   */
  const fitTo = (state) => {
    const total = Math.max(2, (state?.total || FPS * 5) / FPS);
    zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, (width() - GUTTER - LEAD - TAIL) / total));
    scroll = 0;
  };

  /**
   * Zoom around a point on the tracks.
   *
   * @param {number} factor - Multiplier on the pixels per second.
   * @param {number} [aroundX] - The position held still, the middle by default.
   * @returns {void}
   */
  const zoomBy = (factor, aroundX) => {
    manual = true;
    const x = Number.isFinite(aroundX) ? aroundX : (GUTTER + width()) / 2;
    const held = (x - GUTTER - LEAD) / zoom + scroll;
    zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));
    scroll = clampScroll(held - (x - GUTTER - LEAD) / zoom);
    paint();
  };

  /**
   * A scroll position kept inside the run.
   *
   * @param {number} seconds - The asked position.
   * @returns {number} The position, from 0 to a little past the end.
   */
  const clampScroll = (seconds) => {
    const total = (current()?.total ?? 0) / FPS;
    const visible = (width() - GUTTER - LEAD) / zoom;
    return Math.max(0, Math.min(seconds, Math.max(0, total + TAIL / zoom - visible)));
  };

  /**
   * Scroll so one scene is on screen.
   *
   * @param {number} index - The scene, from 0.
   * @returns {void}
   */
  const reveal = (index) => {
    const entry = current()?.layout?.[index];
    if (!entry) return;
    const left = entry.start / FPS;
    const right = entry.end / FPS;
    const visible = (width() - GUTTER - LEAD - 20) / zoom;
    if (left < scroll) scroll = clampScroll(left - 0.5);
    else if (right > scroll + visible) scroll = clampScroll(right - visible + 0.5);
    paint();
  };

  /**
   * The rows as a gesture in progress would have them.
   *
   * @param {object} state - What `current` answered.
   * @returns {object} The same state, laid out again around the gesture.
   */
  const previewed = (state) => {
    if (!drag || !state) return state;
    const rows = state.rows.map((row) => ({ ...row }));
    if (drag.grip === GRIP.END && rows[drag.index]) {
      rows[drag.index].seconds = drag.seconds;
    } else if ((drag.grip === GRIP.SEAM || drag.grip === GRIP.HEAD) && drag.moving && rows[drag.index]) {
      rows[drag.index].overlap = drag.frames;
    } else if (drag.grip === GRIP.CLIP && drag.moving && drag.to !== drag.index) {
      const [moved] = rows.splice(drag.index, 1);
      rows.splice(drag.to, 0, moved);
    } else {
      return state;
    }
    return withLayout({ ...state, rows });
  };

  /**
   * Where a scene's clip is drawn to, short of the frames the next scene cuts away.
   *
   * @param {object} state - The laid out state.
   * @param {number} index - The scene, from 0.
   * @returns {number} The last frame drawn, exclusive.
   */
  const drawnEnd = (state, index) => drawnEndOf(state.layout, index);

  /**
   * A scene's frames drawn along its clip, darkened under the title.
   *
   * @returns {void}
   */
  const paintFilmstrip = (pen, shown, entry, state, index, x0, x1, y, tall) => {
    const fields = shown.sheet.fields;
    const aspect = (Number(fields.cell_width) || 16) / (Number(fields.cell_height) || 9);
    const thumb = Math.max(24, Math.round(tall * aspect));
    const last = drawnEnd(state, index) - 1;
    pen.save();
    pen.globalAlpha = shown.stale ? 0.35 : 0.85;
    for (let x = x0; x < x1; x += thumb) {
      const at = Math.max(entry.start, Math.min(last, frameAt(x + thumb / 2)));
      drawSheetFrame(pen, shown.sheet, sheetFrameAt(entry, fields, at), x, y, thumb, tall);
    }
    pen.globalAlpha = 1;
    const shade = pen.createLinearGradient(0, y, 0, y + tall);
    shade.addColorStop(0, "rgba(0, 0, 0, 0.62)");
    shade.addColorStop(0.5, "rgba(0, 0, 0, 0.2)");
    shade.addColorStop(1, "rgba(0, 0, 0, 0.5)");
    pen.fillStyle = shade;
    pen.fillRect(x0, y, x1 - x0, tall);
    pen.restore();
  };

  /**
   * Where a pinning asset lands in the finished clip.
   *
   * @param {object} asset - The asset.
   * @param {object} entry - The layout entry of its scene.
   * @param {object|null} next - The row after its scene, or null for the last.
   * @param {number} scene - Its scene, from 0.
   * @returns {{at: number, guide: number}|null} The frame and the frames it covers, or null
   *   where it lands outside the scene.
   */
  const pinnedAt = (asset, entry, next, scene) => {
    const guide = asset.frames > 1 ? guideLength(asset.frames) : 1;
    let index;
    if (asset.role === "first frame") index = entry.head;
    else if (asset.role === "last frame") {
      const cut = next ? cutsInto(next.continuity, next.overlap, next.source, scene + 1) : false;
      index = closingIndex(entry.window, guide, cut);
    }
    else index = resolvedIndex(asset.frame, entry.head, entry.window, guide);
    if (index === null || index === undefined) return null;
    return { at: entry.start - entry.head + index, guide };
  };

  /**
   * The scene a frame of the clip falls in.
   *
   * @param {object} state - The laid out state.
   * @param {number} frame - A frame of the clip, fractional.
   * @returns {number} The scene, from 0, or -1 past the end or with no scenes.
   */
  const sceneAt = (state, frame) => {
    for (let index = 0; index < state.layout.length; index += 1) {
      const entry = state.layout[index];
      if (frame >= entry.start && frame < drawnEnd(state, index)) return index;
    }
    return -1;
  };

  /**
   * Where a file dropped at a point would land, and what it would become.
   *
   * @param {object} state - The laid out state.
   * @param {number} x - Across the tracks, in CSS pixels.
   * @param {number} y - Down the tracks, in CSS pixels.
   * @param {string} kind - `picture`, `clip`, `sound`, or empty when it cannot be told.
   * @returns {object} `{valid, action, scene, segment, role, frame, label, lane, x}`.
   */
  const dropTarget = (state, x, y, kind) => {
    const lane = laneAt(y);
    const refused = (label) => ({ valid: false, label, lane, x });
    if (!lane) return refused("");
    const pictured = kind !== "sound";
    const referenceRole = kind === "sound" ? "reference audio"
      : kind === "clip" ? "reference clip" : "reference picture";
    if (lane === "shared") {
      return { valid: true, action: "reference", scene: null, segment: 0, role: referenceRole,
        frame: 0, label: `${ROLE_NAMES[referenceRole]} in every scene`, lane, x };
    }
    if (x < GUTTER) return refused("");
    const frame = frameAt(x);
    const count = state.layout.length;
    const index = sceneAt(state, frame);
    if (lane === "refs") {
      if (!count) return refused("Add a scene first");
      const at = index < 0 ? count - 1 : index;
      return { valid: true, action: "reference", scene: at, segment: at + 1, role: referenceRole,
        frame: 0, label: `${ROLE_NAMES[referenceRole]} in scene ${at + 1}`, lane, x };
    }
    if (index < 0 && (lane === "scenes" || !count)) {
      if (!pictured) return refused("A sound cannot open a scene");
      return { valid: true, action: "new-scene", scene: count, segment: count + 1,
        role: "first frame", frame: 0, label: "New scene opening on this frame", lane, x };
    }
    const at = index < 0 ? count - 1 : index;
    const entry = state.layout[at];
    const end = drawnEnd(state, at);
    if (pictured && Math.abs(x - xOf(entry.start)) <= SNAP_PX) {
      return { valid: true, action: "pin", scene: at, segment: at + 1, role: "first frame",
        frame: 0, label: `Opening frame of scene ${at + 1}`, lane, x: xOf(entry.start) };
    }
    if (pictured && (index < 0 || Math.abs(x - xOf(end)) <= SNAP_PX)) {
      return { valid: true, action: "pin", scene: at, segment: at + 1, role: "last frame",
        frame: 0, label: `Closing frame of scene ${at + 1}`, lane, x: xOf(end) };
    }
    const most = Math.max(0, entry.window - entry.head - 1);
    const into = Math.max(0, Math.min(most, Math.round(frame - entry.start)));
    const what = kind === "sound" ? "Sound" : "Keyframe";
    return { valid: true, action: "pin", scene: at, segment: at + 1, role: "keyframe",
      frame: into, label: `${what} at ${durationOf(into).toFixed(2)}s in scene ${at + 1}`,
      lane, x: xOf(entry.start + into) };
  };

  /**
   * Draw everything.
   *
   * @returns {void}
   */
  const paint = () => {
    if (disposed) return;
    const theme = readTheme();
    const ratio = surfaceRatio(canvas);
    const wide = Math.min(MAX_EDGE, Math.max(1, Math.round(width() * ratio)));
    const tall = Math.min(MAX_EDGE, Math.max(1, Math.round(height() * ratio)));
    if (canvas.width !== wide) canvas.width = wide;
    if (canvas.height !== tall) canvas.height = tall;
    const pen = canvas.getContext("2d");
    if (!pen) return;
    pen.setTransform(ratio, 0, 0, ratio, 0, 0);
    const w = width();
    const h = height();
    pen.clearRect(0, 0, w, h);
    pen.fillStyle = theme.bg;
    pen.fillRect(0, 0, w, h);
    regions = [];

    const base = current();
    if (!base) return;
    // The run stays fitted until it is zoomed by hand, and holds still under a gesture.
    if (!manual && !drag && w > GUTTER + LEAD + TAIL) fitTo(base);
    const state = previewed(base);
    const placed = lanes();
    const dragKind = state.dragging?.kind ?? dropping?.kind ?? null;

    // Hairlines between the tracks, and every track outlined while a file is dragged.
    for (const lane of Object.values(placed)) {
      pen.strokeStyle = theme.border;
      pen.globalAlpha = 0.7;
      pen.beginPath();
      pen.moveTo(GUTTER, Math.round(lane.top + lane.height + LANE_GAP / 2) + 0.5);
      pen.lineTo(w, Math.round(lane.top + lane.height + LANE_GAP / 2) + 0.5);
      pen.stroke();
      pen.globalAlpha = 1;
      if (dragKind) {
        pen.setLineDash([5, 4]);
        pen.strokeStyle = theme.accent;
        pen.globalAlpha = 0.5;
        pen.strokeRect(GUTTER + 0.5, lane.top + 0.5, w - GUTTER - 1, lane.height - 1);
        pen.globalAlpha = 1;
        pen.setLineDash([]);
      }
    }

    pen.save();
    pen.beginPath();
    pen.rect(GUTTER, 0, w - GUTTER, h);
    pen.clip();
    paintRuler(pen, theme, state, w, h);
    paintScenes(pen, theme, state, placed.scenes, w);
    paintPins(pen, theme, state, placed.frames, w);
    paintReferences(pen, theme, state, placed.refs, w);
    paintShared(pen, theme, state, placed.shared, w);
    paintDrop(pen, theme, state, placed, w);
    paintPlayhead(pen, theme, state, h);
    pen.restore();

    paintGutter(pen, theme, state, placed, h, dragKind);
    paintBar(pen, theme, state, w, h);
    paintReadout(pen, theme, w);
  };

  /**
   * The ruler and its grid lines.
   *
   * @returns {void}
   */
  const paintRuler = (pen, theme, state, w, h) => {
    pen.fillStyle = theme.bgLight;
    pen.fillRect(GUTTER, 0, w - GUTTER, RULER);
    pen.strokeStyle = theme.border;
    pen.beginPath();
    pen.moveTo(GUTTER, RULER + 0.5);
    pen.lineTo(w, RULER + 0.5);
    pen.stroke();
    const visible = (w - GUTTER - LEAD) / zoom;
    const step = tickStep(visible);
    const minor = step / (step >= 1 && step <= 2 ? 4 : 5);
    pen.lineWidth = 1;
    const first = Math.floor(scroll / minor) * minor;
    for (let mark = first; mark <= scroll + visible + step; mark += minor) {
      const x = Math.round(GUTTER + LEAD + (mark - scroll) * zoom) + 0.5;
      if (x < GUTTER) continue;
      const major = Math.abs(mark / step - Math.round(mark / step)) < 1e-6;
      pen.strokeStyle = major ? theme.fgMuted : theme.border;
      pen.beginPath();
      pen.moveTo(x, major ? RULER - 9 : RULER - 4);
      pen.lineTo(x, RULER);
      pen.stroke();
      if (major) {
        pen.strokeStyle = theme.border;
        pen.globalAlpha = 0.35;
        pen.beginPath();
        pen.moveTo(x, RULER);
        pen.lineTo(x, h - BAR_HEIGHT - 6);
        pen.stroke();
        pen.globalAlpha = 1;
        pen.fillStyle = theme.fgMuted;
        pen.font = `11px ${SANS}`;
        pen.textAlign = "left";
        pen.textBaseline = "top";
        pen.fillText(rulerLabel(Math.round(mark * 1000) / 1000, step), x + 4, 5);
      }
    }
    regions.push({ kind: GRIP.PLAYHEAD, x: GUTTER, y: 0, w: w - GUTTER, h: RULER, cursor: "col-resize",
      title: "Drag to move the playhead" });
  };

  /**
   * The scenes track: one clip per scene, the transition between each pair, and the add button.
   *
   * @returns {void}
   */
  const paintScenes = (pen, theme, state, lane, w) => {
    const y = lane.top + 2;
    const tall = lane.height - 4;
    const clips = [];
    state.layout.forEach((entry, index) => {
      const row = state.rows[index];
      const x0 = xOf(entry.start);
      const x1 = Math.max(x0 + 22, xOf(Math.max(drawnEnd(state, index), entry.start + 1)));
      const selected = state.selected === index;
      const hovered = hover?.index === index && (hover.kind === GRIP.CLIP || hover.kind === GRIP.END);
      const moving = drag?.grip === GRIP.CLIP && drag.moving && drag.to === index;
      const tint = row.colour ?? sceneTint(index, row.continuity, row.overlap, row.sound);
      const base = tint.stripe;
      clips.push({ index, x0, x1 });

      trace(pen, x0 + 1, y, x1 - x0 - 2, tall, RADIUS);
      pen.fillStyle = theme.panelBg;
      pen.fill();
      pen.fillStyle = base;
      pen.globalAlpha = moving ? 0.5 : selected ? 0.42 : hovered ? 0.32 : 0.24;
      pen.fill();
      pen.globalAlpha = 1;
      pen.save();
      pen.clip();
      const shown = row.preview;
      const onFilm = Boolean(shown?.sheet);
      if (onFilm) paintFilmstrip(pen, shown, entry, state, index, x0, x1, y, tall);
      pen.fillStyle = base;
      pen.fillRect(x0, y, x1 - x0, 3);
      if (shown?.live) {
        // How far the scene being sampled has got.
        const done = Math.max(0, Math.min(1, shown.live.step / Math.max(1, shown.live.steps)));
        pen.fillStyle = theme.accent;
        pen.fillRect(x0, y + tall - 4, (x1 - x0) * done, 4);
      }

      // Carried frames, striped along the foot of the scene they come from.
      if (index > 0 && entry.head > 0 && entry.trimmed <= 0) {
        const hx0 = xOf(entry.start - entry.head);
        pen.restore();
        pen.save();
        pen.beginPath();
        pen.rect(hx0, y + tall - 12, Math.max(2, x0 - hx0), 12);
        pen.clip();
        pen.fillStyle = base;
        pen.globalAlpha = 0.25;
        pen.fillRect(hx0, y + tall - 12, x0 - hx0, 12);
        pen.globalAlpha = 0.7;
        pen.strokeStyle = base;
        for (let s = hx0 - 12; s < x0 + 2; s += 5) {
          pen.beginPath();
          pen.moveTo(s, y + tall);
          pen.lineTo(s + 12, y + tall - 12);
          pen.stroke();
        }
        pen.globalAlpha = 1;
        pen.restore();
        regions.push({
          kind: GRIP.HEAD, index, x: hx0 - GRAB, y: y + tall - 14, w: GRAB * 2, h: 14,
          cursor: "ew-resize", title: `${entry.head} frames carried from the scene before. Drag to change`,
        });
        pen.save();
        trace(pen, x0 + 1, y, x1 - x0 - 2, tall, RADIUS);
        pen.clip();
      }

      // The number, the title and the length.
      const left = x0 + (index > 0 ? SEAM_RADIUS + 8 : 10);
      pen.fillStyle = base;
      trace(pen, left, y + 9, 20, 18, 5);
      pen.fill();
      pen.fillStyle = tint.text;
      pen.font = `600 11px ${SANS}`;
      pen.textAlign = "center";
      pen.textBaseline = "middle";
      pen.fillText(String(index + 1), left + 10, y + 18.5);
      pen.textAlign = "left";
      pen.fillStyle = onFilm ? "#ffffff" : theme.fg;
      pen.font = `600 12px ${SANS}`;
      pen.fillText(ellipsized(pen, row.title || "Untitled scene", x1 - left - 34 - (row.warning ? 16 : 0)), left + 26, y + 18.5);
      pen.font = `11px ${SANS}`;
      pen.fillStyle = onFilm ? "rgba(255, 255, 255, 0.82)" : theme.fgMuted;
      const onScreen = durationOf(drawnEnd(state, index) - entry.start);
      const parts = [`${onScreen.toFixed(2)}s`];
      if (shown?.live) parts.unshift(`step ${shown.live.step}/${shown.live.steps}`);
      else if (shown?.stale) parts.unshift("changed");
      else if (shown?.partial) parts.unshift(`stopped ${shown.sheet.fields.step}/${shown.sheet.fields.steps}`);
      if (entry.source !== null && entry.source !== undefined && index > 0 && entry.source !== index - 1) {
        parts.push(`from scene ${entry.source + 1}`);
      }
      if (row.model && row.model !== "auto") parts.push(row.model);
      pen.fillText(ellipsized(pen, parts.join("  ·  "), x1 - left - SEAM_RADIUS - 8), left, y + 39);
      if (row.warning) {
        pen.fillStyle = theme.warning;
        pen.font = `600 13px ${SANS}`;
        pen.textAlign = "right";
        pen.fillText("⚠", x1 - 12, y + 18.5);
        pen.textAlign = "left";
      }
      pen.restore();

      // The outline, and the trim handle on the right edge.
      trace(pen, x0 + 1.5, y + 0.5, x1 - x0 - 3, tall - 1, RADIUS);
      pen.lineWidth = selected ? 2 : 1;
      pen.strokeStyle = selected ? base : theme.border;
      pen.stroke();
      pen.lineWidth = 1;
      if (selected || hovered) {
        pen.fillStyle = selected ? base : theme.fgMuted;
        trace(pen, x1 - 8, y + 7, 4, 20, 2);
        pen.fill();
      }
      regions.push({
        kind: GRIP.END, index, x: x1 - GRAB - 4, y, w: GRAB + 8, h: tall, cursor: "ew-resize",
        title: `${onScreen.toFixed(2)}s on screen, ${durationOf(snapClip(entry.frames)).toFixed(2)}s sampled. `
          + "Drag to change the length",
      });
    });

    // Transition buttons, drawn over the seams so they sit above both clips.
    state.layout.forEach((entry, index) => {
      if (index === 0) return;
      const row = state.rows[index];
      const effective = effectiveTransition(row.continuity, row.overlap, row.sound);
      const bridged = resolvedTransition(row.continuity, row.sound).bridged;
      const shown = TRANSITIONS[effective];
      const cx = xOf(entry.start);
      const cy = y + tall - SEAM_RADIUS - 4;
      const lit = hover?.kind === GRIP.SEAM && hover.index === index;
      pen.beginPath();
      pen.arc(cx, cy, SEAM_RADIUS, 0, Math.PI * 2);
      pen.fillStyle = lit ? theme.accent : theme.bgLight;
      pen.fill();
      // The ring keeps the transition's colour where the scene has a row colour of its own.
      pen.lineWidth = 2;
      pen.strokeStyle = lit ? theme.accent : sceneTint(index, row.continuity, row.overlap, row.sound).stripe;
      pen.stroke();
      pen.lineWidth = 1;
      pen.fillStyle = lit ? theme.selectionText : theme.fg;
      pen.font = `600 12px ${SANS}`;
      pen.textAlign = "center";
      pen.textBaseline = "middle";
      pen.fillText(shown.glyph, cx, cy + 0.5);
      pen.textAlign = "left";
      if (bridged && !BRIDGING.includes(effective)) {
        // The sound runs on under this cut picture.
        pen.font = `600 10px ${SANS}`;
        pen.textAlign = "center";
        pen.fillStyle = theme.fg;
        pen.fillText("\u266A", cx + SEAM_RADIUS + 1, cy - SEAM_RADIUS + 2);
        pen.textAlign = "left";
      }
      const carried = bridged ? `, ${bridgeFrames(row.overlap)} frames of sound`
        : CUT_LIKE.includes(effective) || effective === "cut" ? "" : `, ${entry.head} frames carried`;
      const named = shown.name;
      regions.unshift({
        kind: GRIP.SEAM, index, x: cx - SEAM_RADIUS - 2, y: cy - SEAM_RADIUS - 2,
        w: SEAM_RADIUS * 2 + 4, h: SEAM_RADIUS * 2 + 4, cursor: "pointer",
        title: `Into scene ${index + 1}: ${named}${carried}. Click to change, drag to set the overlap`,
      });
    });

    // The add button past the last clip, or the first scene's invitation.
    const ax = clips.length ? clips[clips.length - 1].x1 + 10 : xOf(0);
    const aw = clips.length ? ADD_WIDTH : Math.min(360, Math.max(200, w - GUTTER - LEAD - 20));
    const lit = hover?.kind === "add";
    trace(pen, ax + 0.5, y + 0.5, aw - 1, tall - 1, RADIUS);
    pen.setLineDash([5, 4]);
    pen.strokeStyle = lit ? theme.accent : theme.border;
    pen.stroke();
    pen.setLineDash([]);
    if (lit) {
      pen.fillStyle = theme.accentBg;
      pen.fill();
    }
    pen.fillStyle = lit ? theme.accent : theme.fgMuted;
    pen.font = `600 12px ${SANS}`;
    pen.textAlign = "center";
    pen.textBaseline = "middle";
    pen.fillText(clips.length ? "+  Add scene" : "+  Add the first scene", ax + aw / 2, y + tall / 2);
    pen.textAlign = "left";
    regions.push({ kind: "add", x: ax, y, w: aw, h: tall, cursor: "pointer",
      title: "Add a scene at the end. A picture dropped here opens it" });

    // A row the nodes refuse to follow.
    if (state.refused !== null) {
      pen.fillStyle = theme.warning;
      pen.font = `12px ${SANS}`;
      pen.textBaseline = "middle";
      pen.fillText(`⚠ Scene ${state.refused + 1} cannot continue from the scene it names`,
        ax + aw + 12, y + tall / 2, Math.max(0, w - ax - aw - 20));
    }

    // Clips are hit tested after their handles, so a handle wins at an edge.
    for (const clip of clips) {
      regions.push({
        kind: GRIP.CLIP, index: clip.index, x: clip.x0, y, w: clip.x1 - clip.x0, h: tall,
        cursor: "grab", title: "Click to edit, drag to reorder, double click to write its prompt",
      });
    }
  };

  /**
   * The pinned frames track: each opening frame, closing frame and keyframe at its place.
   *
   * @returns {void}
   */
  const paintPins = (pen, theme, state, lane, w) => {
    const y = lane.top + 4;
    let any = false;
    const labels = [];
    state.layout.forEach((entry, index) => {
      const own = state.assets.filter((asset) => asset.segment === index + 1
        && (asset.role === "first frame" || asset.role === "last frame" || asset.role === "keyframe"));
      for (const asset of own) {
        any = true;
        let placedAt = null;
        try {
          placedAt = pinnedAt(asset, entry, state.layout[index + 1] ? state.rows[index + 1] : null, index);
        } catch (error) {
          console.error(`[${logName}] An asset could not be placed:`, error);
        }
        const outside = placedAt === null;
        let at = outside ? entry.start : placedAt.at;
        const guide = outside ? 1 : placedAt.guide;
        if (drag?.grip === GRIP.PIN && drag.id === asset.id && drag.moved) at = entry.start + drag.frame;
        const mx = xOf(at);
        let px = mx - PIN_WIDTH / 2;
        if (asset.role === "first frame") px = mx + 1;
        else if (asset.role === "last frame") px = xOf(at + guide) - PIN_WIDTH - 1;
        const chosen = state.selectedAsset === asset.id;
        const lit = hover?.kind === GRIP.PIN && hover.id === asset.id;

        // The span a pinned clip covers.
        if (guide > 1) {
          pen.fillStyle = theme.accent;
          pen.globalAlpha = 0.35;
          pen.fillRect(mx, y + PIN_HEIGHT + 3, Math.max(2, xOf(at + guide) - mx), 3);
          pen.globalAlpha = 1;
        }
        // The stem from the scene above to the frame.
        pen.strokeStyle = outside ? theme.warning : theme.accent;
        pen.beginPath();
        pen.moveTo(Math.round(mx) + 0.5, lane.top - LANE_GAP);
        pen.lineTo(Math.round(mx) + 0.5, y);
        pen.stroke();

        trace(pen, px, y, PIN_WIDTH, PIN_HEIGHT, 5);
        pen.fillStyle = theme.bgLight;
        pen.fill();
        const picture = thumbnail(asset.thumb);
        pen.save();
        trace(pen, px, y, PIN_WIDTH, PIN_HEIGHT, 4);
        pen.clip();
        if (picture) {
          drawCovered(pen, picture, px, y, PIN_WIDTH, PIN_HEIGHT);
        } else {
          pen.fillStyle = theme.fgMuted;
          pen.font = `16px ${SANS}`;
          pen.textAlign = "center";
          pen.textBaseline = "middle";
          pen.fillText(kindGlyph(asset.kind), px + PIN_WIDTH / 2, y + PIN_HEIGHT / 2);
          pen.textAlign = "left";
        }
        pen.restore();
        trace(pen, px + 0.5, y + 0.5, PIN_WIDTH - 1, PIN_HEIGHT - 1, 4);
        pen.lineWidth = chosen || lit ? 2 : 1;
        pen.strokeStyle = outside ? theme.warning : chosen ? theme.accent : lit ? theme.fgMuted : theme.border;
        pen.stroke();
        pen.lineWidth = 1;

        const word = asset.role === "first frame" ? "Opening"
          : asset.role === "last frame" ? "Closing"
          : `${durationOf(Math.max(0, at - entry.start)).toFixed(2)}s`;
        labels.push({ text: outside ? "outside" : word, x: px + PIN_WIDTH / 2, warn: outside });
        const role = ROLE_NAMES[asset.role] ?? asset.role;
        regions.unshift({
          kind: GRIP.PIN, id: asset.id, index, x: px, y, w: PIN_WIDTH, h: PIN_HEIGHT + 16,
          cursor: asset.role === "keyframe" ? "ew-resize" : "pointer",
          title: `${role} of scene ${index + 1}: ${asset.name}`
            + (outside ? ". It lands outside the scene" : "")
            + (asset.role === "keyframe" ? ". Drag to move it" : ""),
          draggable: asset.role === "keyframe",
        });
      }
    });
    // Labels left to right; one that would land on the label before it is left to the tooltip.
    pen.font = `10px ${SANS}`;
    pen.textAlign = "center";
    pen.textBaseline = "top";
    let clear = -Infinity;
    for (const label of labels.sort((a, b) => a.x - b.x)) {
      const text = ellipsized(pen, label.text, PIN_WIDTH + 16);
      const half = pen.measureText(text).width / 2;
      if (label.x - half < clear + LABEL_GAP) continue;
      pen.fillStyle = label.warn ? theme.warning : theme.fgMuted;
      pen.fillText(text, label.x, y + PIN_HEIGHT + 6);
      clear = label.x + half;
    }
    pen.textAlign = "left";
    if (!any) laneHint(pen, theme, lane, w, "Drop a picture to pin it");
    regions.push({ kind: "lane", lane: lane.id, x: GUTTER, y: lane.top, w: w - GUTTER, h: lane.height,
      cursor: "default", title: "" });
  };

  /**
   * One reference chip.
   *
   * @returns {number} The chip's width.
   */
  const chip = (pen, theme, state, asset, x, y, most) => {
    const text = `${asset.tag && asset.tag !== asset.role ? `${asset.tag} ` : ""}${asset.name}`;
    pen.font = `11px ${SANS}`;
    const wide = Math.min(most, CHIP_MAX, CHIP_THUMB + 14 + pen.measureText(text).width);
    if (wide < CHIP_THUMB + 8) return 0;
    const chosen = state.selectedAsset === asset.id;
    const lit = hover?.kind === "ref" && hover.id === asset.id;
    trace(pen, x, y, wide, CHIP_HEIGHT, CHIP_HEIGHT / 2);
    pen.fillStyle = theme.bgLight;
    pen.fill();
    pen.lineWidth = chosen || lit ? 2 : 1;
    pen.strokeStyle = chosen ? theme.accent : lit ? theme.fgMuted : theme.border;
    pen.stroke();
    pen.lineWidth = 1;
    const picture = thumbnail(asset.thumb);
    pen.save();
    pen.beginPath();
    pen.arc(x + CHIP_HEIGHT / 2, y + CHIP_HEIGHT / 2, CHIP_THUMB / 2, 0, Math.PI * 2);
    pen.clip();
    if (picture) {
      drawCovered(pen, picture, x + 3, y + 3, CHIP_THUMB, CHIP_THUMB);
    } else {
      pen.fillStyle = theme.bgLight;
      pen.fillRect(x + 3, y + 3, CHIP_THUMB, CHIP_THUMB);
      pen.fillStyle = theme.fgMuted;
      pen.font = `10px ${SANS}`;
      pen.textAlign = "center";
      pen.textBaseline = "middle";
      pen.fillText(kindGlyph(asset.kind), x + CHIP_HEIGHT / 2, y + CHIP_HEIGHT / 2);
      pen.textAlign = "left";
    }
    pen.restore();
    pen.fillStyle = theme.fg;
    pen.font = `11px ${SANS}`;
    pen.textBaseline = "middle";
    pen.fillText(ellipsized(pen, text, wide - CHIP_THUMB - 15), x + CHIP_THUMB + 9, y + CHIP_HEIGHT / 2 + 0.5);
    return wide;
  };

  /**
   * The references track: each scene's referenced pictures, clips and sounds under it.
   *
   * @returns {void}
   */
  const paintReferences = (pen, theme, state, lane, w) => {
    const y = lane.top + (lane.height - CHIP_HEIGHT) / 2;
    let any = false;
    state.layout.forEach((entry, index) => {
      const own = state.assets.filter((asset) => asset.segment === index + 1
        && String(asset.role).startsWith("reference"));
      if (!own.length) return;
      any = true;
      const left = xOf(entry.start) + 4;
      const right = xOf(drawnEnd(state, index)) - 4;
      let x = left;
      own.forEach((asset, at) => {
        const rest = own.length - at;
        const room = right - x - (rest > 1 ? 34 : 0);
        const used = room > CHIP_THUMB + 20 ? chip(pen, theme, state, asset, x, y, room) : 0;
        if (!used) {
          if (at < own.length && x < right) {
            pen.fillStyle = theme.fgMuted;
            pen.font = `600 11px ${SANS}`;
            pen.textBaseline = "middle";
            pen.fillText(`+${rest}`, x + 2, y + CHIP_HEIGHT / 2);
          }
          x = right + 1;
          return;
        }
        regions.unshift({
          kind: "ref", id: asset.id, index, x, y, w: used, h: CHIP_HEIGHT, cursor: "pointer",
          title: `${ROLE_NAMES[asset.role] ?? asset.role} in scene ${index + 1}, named ${asset.tag} `
            + `in its prompt: ${asset.name}. Drag to another scene`,
        });
        x += used + 4;
      });
    });
    if (!any) laneHint(pen, theme, lane, w, "Drop a reference for one scene");
    regions.push({ kind: "lane", lane: lane.id, x: GUTTER, y: lane.top, w: w - GUTTER, h: lane.height,
      cursor: "default", title: "" });
  };

  /**
   * The all-scenes track: references every scene's prompt can name.
   *
   * @returns {void}
   */
  const paintShared = (pen, theme, state, lane, w) => {
    const y = lane.top + (lane.height - CHIP_HEIGHT) / 2;
    const shared = state.assets.filter((asset) => asset.segment === 0);
    if (!shared.length) {
      laneHint(pen, theme, lane, w, "Drop a reference for every scene");
    } else {
      const span = Math.max(xOf(state.total), xOf(0) + 60);
      pen.fillStyle = theme.accent;
      pen.globalAlpha = 0.14;
      trace(pen, xOf(0), lane.top + 3, span - xOf(0), lane.height - 6, RADIUS);
      pen.fill();
      pen.globalAlpha = 1;
      let x = Math.max(xOf(0), GUTTER) + 4;
      for (const asset of shared) {
        const used = chip(pen, theme, state, asset, x, y, w - x - 8);
        if (!used) break;
        regions.unshift({
          kind: "ref", id: asset.id, index: null, x, y, w: used, h: CHIP_HEIGHT, cursor: "pointer",
          title: `${ROLE_NAMES[asset.role] ?? asset.role} in every scene, named ${asset.tag}: ${asset.name}. `
            + "Drag to one scene",
        });
        x += used + 4;
      }
    }
    regions.push({ kind: "lane", lane: lane.id, x: GUTTER, y: lane.top, w: w - GUTTER, h: lane.height,
      cursor: "default", title: "" });
  };

  /**
   * A quiet sentence in an empty track.
   *
   * @returns {void}
   */
  const laneHint = (pen, theme, lane, w, text) => {
    pen.fillStyle = theme.fgMuted;
    pen.globalAlpha = 0.85;
    pen.font = `italic 11px ${SANS}`;
    pen.textAlign = "left";
    pen.textBaseline = "middle";
    pen.fillText(ellipsized(pen, text, w - GUTTER - LEAD - 10), GUTTER + LEAD + 2, lane.top + lane.height / 2);
    pen.globalAlpha = 1;
  };

  /**
   * Where a file being dragged would land.
   *
   * @returns {void}
   */
  const paintDrop = (pen, theme, state, placed, w) => {
    if (!dropping?.target?.lane) return;
    const target = dropping.target;
    const lane = placed[target.lane];
    if (!lane) return;
    let x0 = GUTTER;
    let x1 = w;
    if (target.valid && target.scene !== null && target.scene !== undefined && state.layout[target.scene]) {
      x0 = xOf(state.layout[target.scene].start);
      x1 = xOf(drawnEnd(state, target.scene));
    } else if (target.valid && target.action === "new-scene") {
      x0 = xOf(state.total);
      x1 = x0 + ADD_WIDTH;
    }
    if (target.valid && target.lane !== "shared") {
      pen.fillStyle = theme.accent;
      pen.globalAlpha = 0.16;
      pen.fillRect(x0, lane.top, x1 - x0, lane.height);
      pen.globalAlpha = 1;
    } else if (target.valid) {
      pen.fillStyle = theme.accent;
      pen.globalAlpha = 0.16;
      pen.fillRect(GUTTER, lane.top, w - GUTTER, lane.height);
      pen.globalAlpha = 1;
    }
    if (target.valid && target.action === "pin") {
      pen.strokeStyle = theme.accent;
      pen.lineWidth = 2;
      pen.beginPath();
      pen.moveTo(target.x, placed.scenes.top);
      pen.lineTo(target.x, lane.top + lane.height);
      pen.stroke();
      pen.lineWidth = 1;
    }
    if (target.label) pill(pen, theme, target.label, dropping.x, lane.top - 4, w, !target.valid);
  };

  /**
   * A label in a dark pill above a point.
   *
   * @returns {void}
   */
  const pill = (pen, theme, text, x, bottom, w, warn = false) => {
    pen.font = `600 12px ${SANS}`;
    const wide = pen.measureText(text).width + 18;
    const px = Math.max(GUTTER + 4, Math.min(w - wide - 4, x - wide / 2));
    const py = Math.max(2, bottom - 26);
    pen.save();
    pen.shadowColor = theme.shadow;
    pen.shadowBlur = 12;
    pen.shadowOffsetY = 4;
    trace(pen, px, py, wide, 24, 6);
    pen.fillStyle = theme.bgLight;
    pen.fill();
    pen.restore();
    trace(pen, px + 0.5, py + 0.5, wide - 1, 23, 6);
    pen.strokeStyle = warn ? theme.warning : theme.border;
    pen.stroke();
    pen.fillStyle = warn ? theme.warning : theme.fg;
    pen.textAlign = "left";
    pen.textBaseline = "middle";
    pen.fillText(text, px + 9, py + 12.5);
  };

  /**
   * The playhead across every track.
   *
   * @returns {void}
   */
  const paintPlayhead = (pen, theme, state, h) => {
    const px = Math.round(xOf(state.playhead * FPS)) + 0.5;
    if (px < GUTTER) return;
    pen.strokeStyle = theme.error;
    pen.lineWidth = 1.5;
    pen.beginPath();
    pen.moveTo(px, RULER - 2);
    pen.lineTo(px, h - BAR_HEIGHT - 6);
    pen.stroke();
    pen.lineWidth = 1;
    pen.fillStyle = theme.error;
    trace(pen, px - 6, 2, 12, RULER - 10, 3);
    pen.fill();
    pen.beginPath();
    pen.moveTo(px - 6, RULER - 9);
    pen.lineTo(px + 6, RULER - 9);
    pen.lineTo(px, RULER - 2);
    pen.closePath();
    pen.fill();
  };

  /**
   * The track names down the left, and the playhead's time above them.
   *
   * @returns {void}
   */
  const paintGutter = (pen, theme, state, placed, h, dragKind) => {
    pen.fillStyle = theme.bgLight;
    pen.fillRect(0, 0, GUTTER, h);
    pen.strokeStyle = theme.border;
    pen.beginPath();
    pen.moveTo(GUTTER - 0.5, 0);
    pen.lineTo(GUTTER - 0.5, h);
    pen.stroke();
    pen.beginPath();
    pen.moveTo(0, RULER + 0.5);
    pen.lineTo(GUTTER, RULER + 0.5);
    pen.stroke();
    pen.fillStyle = theme.fg;
    pen.font = `600 12px ${MONO}`;
    pen.textAlign = "left";
    pen.textBaseline = "middle";
    pen.fillText(clockOf(state.playhead), 12, RULER / 2);
    const hints = {
      scenes: "One prompt each",
      frames: "Held at a moment",
      refs: "Named in a prompt",
      shared: "Named in every prompt",
    };
    for (const lane of Object.values(placed)) {
      const target = dropping?.target?.valid && dropping.target.lane === lane.id;
      pen.fillStyle = target ? theme.accent : dragKind && lane.id !== "scenes" ? theme.fg : theme.fgMuted;
      pen.font = `700 10px ${SANS}`;
      pen.letterSpacing = "0.6px";
      pen.fillText(ellipsized(pen, lane.label.toUpperCase(), GUTTER - 18), 12, lane.top + lane.height / 2);
      pen.letterSpacing = "0px";
      regions.push({ kind: "header", lane: lane.id, x: 0, y: lane.top, w: GUTTER, h: lane.height,
        cursor: "default", title: hints[lane.id] });
    }
  };

  /**
   * The scroll bar along the bottom, drawn when the run is wider than the view.
   *
   * @returns {void}
   */
  const paintBar = (pen, theme, state, w, h) => {
    const total = state.total / FPS + TAIL / zoom;
    const visible = (w - GUTTER - LEAD) / zoom;
    if (total <= visible + 1e-6) return;
    const track = w - GUTTER - 16;
    const thumb = Math.max(30, (track * visible) / total);
    const x = GUTTER + 8 + ((track - thumb) * scroll) / Math.max(1e-6, total - visible);
    const y = h - BAR_HEIGHT - 3;
    trace(pen, x, y, thumb, BAR_HEIGHT - 2, (BAR_HEIGHT - 2) / 2);
    pen.fillStyle = hover?.kind === GRIP.BAR || drag?.grip === GRIP.BAR ? theme.fgMuted : theme.scrollbarThumb;
    pen.fill();
    regions.unshift({ kind: GRIP.BAR, x, y: y - 2, w: thumb, h: BAR_HEIGHT + 2, cursor: "grab",
      title: "", track, thumb, span: total - visible });
  };

  /**
   * What a drag is about to write, in a pill near the pointer.
   *
   * @returns {void}
   */
  const paintReadout = (pen, theme, w) => {
    if (!drag?.readout) return;
    pill(pen, theme, drag.readout, drag.x, RULER + 26, w);
  };

  /**
   * The region under a point.
   *
   * @param {number} clientX - Across the page.
   * @param {number} clientY - Down the page.
   * @returns {{region: object|null, x: number, y: number}} The region and the position.
   */
  const regionAt = (clientX, clientY) => {
    const box = canvas.getBoundingClientRect();
    const x = (clientX - box.left) * (width() / Math.max(1, box.width));
    const y = (clientY - box.top) * (height() / Math.max(1, box.height));
    for (const region of regions) {
      if (x >= region.x && x <= region.x + region.w && y >= region.y && y <= region.y + region.h) {
        return { region, x, y };
      }
    }
    return { region: null, x, y };
  };

  /**
   * The place a moved clip would drop at.
   *
   * @param {object} state - What `current` answered.
   * @param {number} x - Across the tracks.
   * @returns {number} The index it would land at.
   */
  const dropIndex = (state, x) => {
    const frame = frameAt(x);
    let to = state.layout.length - 1;
    for (let index = 0; index < state.layout.length; index += 1) {
      const entry = state.layout[index];
      if (frame < (entry.start + entry.end) / 2) {
        to = index;
        break;
      }
    }
    return Math.max(0, Math.min(state.layout.length - 1, to));
  };

  /**
   * The scene a dragged reference would move to.
   *
   * @param {object} state - What `current` answered.
   * @param {number} x - Across the tracks.
   * @param {number} y - Down the tracks.
   * @returns {number|null} The scene from 1, 0 for every scene, or null over no scene.
   */
  const referenceTarget = (state, x, y) => {
    const lane = laneAt(y);
    if (lane === "shared") return 0;
    if (!lane) return null;
    const frame = frameAt(x);
    const index = state.layout.findIndex((entry, at) => entry && frame >= entry.start && frame < drawnEnd(state, at));
    return index >= 0 ? index + 1 : null;
  };

  /**
   * Move the playhead under a point.
   *
   * @param {number} x - Across the tracks.
   * @returns {void}
   */
  const scrub = (x) => {
    onChange({ type: "playhead", seconds: Math.max(0, frameAt(Math.max(x, GUTTER + LEAD)) / FPS) });
  };

  canvas.addEventListener("pointerdown", (event) => {
    root.focus({ preventScroll: true });
    const { region, x } = regionAt(event.clientX, event.clientY);
    if (event.button === 2) {
      if (!region) return;
      if (region.kind === GRIP.CLIP || region.kind === GRIP.END || region.kind === GRIP.HEAD) {
        onSelect(region.index);
        onMenu(region.index, event);
      } else if (region.kind === GRIP.SEAM) {
        onTransitionMenu(region.index, event);
      } else if (region.kind === GRIP.PIN || region.kind === "ref") {
        onSelectAsset(region.id);
        onAssetMenu(region.id, event);
      }
      return;
    }
    if (event.button !== 0) return;
    const state = current();
    if (!state || !region) return;
    if (region.kind === GRIP.PLAYHEAD || region.kind === "lane") {
      drag = { grip: GRIP.PLAYHEAD, x };
      scrub(x);
    } else if (region.kind === GRIP.END) {
      const entry = state.layout[region.index];
      onSelect(region.index);
      drag = { grip: GRIP.END, index: region.index, x, seconds: state.rows[region.index].seconds,
        startX: xOf(entry.start - entry.head), readout: "" };
    } else if (region.kind === GRIP.SEAM || region.kind === GRIP.HEAD) {
      const entry = state.layout[region.index];
      drag = { grip: region.kind, index: region.index, x, downX: x, moving: region.kind === GRIP.HEAD,
        frames: state.rows[region.index].overlap, edgeX: xOf(entry.start),
        continuity: state.rows[region.index].continuity, readout: "" };
    } else if (region.kind === GRIP.PIN) {
      onSelectAsset(region.id);
      if (!region.draggable) return;
      const asset = state.assets.find((each) => each.id === region.id);
      drag = { grip: GRIP.PIN, id: region.id, index: region.index, x, downX: x, moved: false,
        frame: Number(asset?.frame) || 0, entry: state.layout[region.index], asset, readout: "" };
    } else if (region.kind === "ref") {
      onSelectAsset(region.id);
      drag = { grip: "ref", id: region.id, index: region.index, x, downX: x, downY: regionAt(event.clientX, event.clientY).y,
        moved: false, segment: null, readout: "" };
    } else if (region.kind === GRIP.CLIP) {
      onSelect(region.index);
      drag = { grip: GRIP.CLIP, index: region.index, to: region.index, x, downX: x, moving: false,
        readout: "" };
    } else if (region.kind === "add") {
      onChange({ type: "add" });
      return;
    } else if (region.kind === GRIP.BAR) {
      drag = { grip: GRIP.BAR, x, downX: x, scroll, region };
    } else {
      return;
    }
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.error(`[${logName}] Failed to capture the pointer:`, error);
    }
    event.preventDefault();
    paint();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!drag) {
      const { region } = regionAt(event.clientX, event.clientY);
      const next = region ? { kind: region.kind, index: region.index, id: region.id } : null;
      const key = (value) => (value ? `${value.kind}:${value.index ?? ""}:${value.id ?? ""}` : "");
      if (key(next) !== key(hover)) {
        hover = next;
        canvas.style.cursor = region?.cursor ?? "default";
        canvas.title = region?.title ?? "";
        paint();
      }
      return;
    }
    if ((event.buttons & 1) === 0) {
      release(event);
      return;
    }
    const { x, y } = regionAt(event.clientX, event.clientY);
    drag.x = x;
    if (drag.grip === GRIP.PLAYHEAD) {
      scrub(x);
      return;
    }
    if (drag.grip === GRIP.BAR) {
      const { track, thumb, span } = drag.region;
      scroll = clampScroll(drag.scroll + ((x - drag.downX) / Math.max(1, track - thumb)) * span);
    } else if (drag.grip === GRIP.END) {
      const asked = Math.max(MIN_SECONDS, Math.min(MAX_SECONDS, (frameAt(x) - frameAt(drag.startX)) / FPS));
      const snapped = snapClip(framesOf(asked));
      drag.seconds = durationOf(snapped);
      drag.readout = `${drag.seconds.toFixed(2)}s  (${snapped} frames)`;
    } else if (drag.grip === GRIP.SEAM || drag.grip === GRIP.HEAD) {
      if (!drag.moving && Math.abs(x - drag.downX) > MOVE_THRESHOLD) drag.moving = true;
      if (drag.moving) {
        const pulled = Math.round(frameAt(drag.edgeX) - frameAt(x));
        const named = drag.continuity;
        if (pulled <= 2) {
          drag.frames = 0;
          drag.readout = "Hard cut, nothing carried";
        } else if (BRIDGING.includes(named)) {
          drag.frames = bridgeFrames(pulled);
          drag.readout = `${drag.frames} frames of sound carried (${durationOf(drag.frames)}s)`;
        } else {
          drag.frames = snapOverlap(pulled);
          drag.readout = `${drag.frames} frames carried (${durationOf(drag.frames)}s)`;
        }
      }
    } else if (drag.grip === GRIP.PIN) {
      if (!drag.moved && Math.abs(x - drag.downX) > MOVE_THRESHOLD) drag.moved = true;
      if (drag.moved) {
        const entry = drag.entry;
        const guide = drag.asset?.frames > 1 ? guideLength(drag.asset.frames) : 1;
        const most = Math.max(0, entry.window - entry.head - guide);
        drag.frame = Math.max(0, Math.min(most, Math.round(frameAt(x) - entry.start)));
        drag.readout = `Keyframe at ${durationOf(drag.frame).toFixed(2)}s into scene ${drag.index + 1}`;
      }
    } else if (drag.grip === "ref") {
      if (!drag.moved && Math.abs(x - drag.downX) + Math.abs(y - drag.downY) > MOVE_THRESHOLD) drag.moved = true;
      if (drag.moved) {
        const state = current();
        drag.segment = state ? referenceTarget(state, x, y) : null;
        canvas.style.cursor = drag.segment === null ? "no-drop" : "grabbing";
        drag.readout = drag.segment === null ? "Drop on a scene or on All scenes"
          : drag.segment === 0 ? "Move to every scene" : `Move to scene ${drag.segment}`;
      }
    } else if (drag.grip === GRIP.CLIP) {
      if (!drag.moving && Math.abs(x - drag.downX) > MOVE_THRESHOLD) drag.moving = true;
      if (drag.moving) {
        const state = current();
        if (state) drag.to = dropIndex(state, x);
        drag.readout = drag.to === drag.index ? "" : `Move to position ${drag.to + 1}`;
        canvas.style.cursor = "grabbing";
      }
    }
    paint();
  });

  /**
   * End a gesture, reporting what it changed.
   *
   * @param {PointerEvent|null} event - What ended it.
   * @returns {void}
   */
  const release = (event) => {
    if (!drag) return;
    const done = drag;
    drag = null;
    try {
      if (event && canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    } catch (error) {
      console.error(`[${logName}] Failed to release the pointer:`, error);
    }
    try {
      if (done.grip === GRIP.END) {
        if (done.readout) onChange({ type: "duration", index: done.index, seconds: done.seconds });
      } else if (done.grip === GRIP.SEAM && !done.moving) {
        if (event) onTransitionMenu(done.index, event);
      } else if (done.grip === GRIP.SEAM || done.grip === GRIP.HEAD) {
        if (done.moving) onChange({ type: "overlap", index: done.index, frames: done.frames });
      } else if (done.grip === GRIP.PIN) {
        if (done.moved) onChange({ type: "keyframe", id: done.id, frame: done.frame });
      } else if (done.grip === GRIP.CLIP && done.moving && done.to !== done.index) {
        onChange({ type: "move", from: done.index, to: done.to });
      } else if (done.grip === "ref" && done.moved && done.segment !== null
        && done.segment !== (done.index === null ? 0 : done.index + 1)) {
        onChange({ type: "reference", id: done.id, segment: done.segment });
      }
    } catch (error) {
      console.error(`[${logName}] A timeline change failed:`, error);
    }
    canvas.style.cursor = "default";
    paint();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", () => release(null));
  canvas.addEventListener("lostpointercapture", () => {
    if (drag) release(null);
  });
  canvas.addEventListener("pointerleave", () => {
    if (drag || !hover) return;
    hover = null;
    canvas.style.cursor = "default";
    canvas.title = "";
    paint();
  });
  canvas.addEventListener("dblclick", (event) => {
    const { region } = regionAt(event.clientX, event.clientY);
    if (region?.kind === GRIP.CLIP) onChange({ type: "edit", index: region.index });
  });
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const { x } = regionAt(event.clientX, event.clientY);
    if (event.ctrlKey || event.metaKey) {
      zoomBy(event.deltaY < 0 ? 1.15 : 1 / 1.15, x);
      return;
    }
    const delta = (Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY)
      * (event.deltaMode === 1 ? 16 : 1);
    scroll = clampScroll(scroll + delta / zoom);
    paint();
  }, { passive: false });

  /**
   * Whether a drag holds something the tracks take.
   *
   * @param {DragEvent} event - The drag event.
   * @returns {string|null} The dragged kind, an empty string when unknown, or null for a drag
   *   the tracks do not take.
   */
  const acceptedKind = (event) => {
    const types = Array.from(event.dataTransfer?.types ?? []);
    if (types.includes(DRAG_TYPE)) {
      try {
        return String(read()?.dragging?.kind ?? "");
      } catch (error) {
        console.error(`[${logName}] The timeline accessor failed:`, error);
        return "";
      }
    }
    if (carriesFiles(event)) return draggedKind(event);
    return null;
  };

  root.addEventListener("dragover", (event) => {
    const kind = acceptedKind(event);
    if (kind === null) return;
    event.preventDefault();
    event.stopPropagation();
    const state = current();
    if (!state) return;
    const { x, y } = regionAt(event.clientX, event.clientY);
    const target = dropTarget(state, x, y, kind);
    event.dataTransfer.dropEffect = target.valid ? "copy" : "none";
    dropping = { kind: kind || "picture", target, x };
    paint();
  });
  root.addEventListener("dragleave", (event) => {
    if (root.contains(event.relatedTarget)) return;
    dropping = null;
    paint();
  });
  root.addEventListener("drop", (event) => {
    const kind = acceptedKind(event);
    if (kind === null) return;
    event.preventDefault();
    event.stopPropagation();
    const state = current();
    dropping = null;
    if (!state) return;
    const { x, y } = regionAt(event.clientX, event.clientY);
    const target = dropTarget(state, x, y, kind);
    paint();
    if (!target.valid) return;
    try {
      onDrop(target, event);
    } catch (error) {
      console.error(`[${logName}] A drop failed:`, error);
    }
  });

  const stopWatchingRatio = watchSurfaceRatio(canvas, () => paint());
  const stopTheme = onThemeChange(() => paint());
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => paint()) : null;
  observer?.observe(root);

  return {
    element: root,
    repaint: paint,
    fit,
    zoomBy,
    reveal,
    dispose() {
      if (disposed) return;
      disposed = true;
      observer?.disconnect();
      stopWatchingRatio?.();
      stopTheme?.();
      thumbs.clear();
    },
  };
}
