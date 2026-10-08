/**
 * The timeline editor for the two video loaders.
 *
 * The chosen file plays on the node, and the two handles over the strip write `start` and `end`.
 */

import { api } from "../../scripts/api.js";
import { fetchWithin } from "./interface/request.js";
import { app } from "../../scripts/app.js";
import { selection } from "./interface/frame_timeline.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { drawStandIn, loadPlaceholder } from "./interface/placeholder.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { roundHalfEven } from "./interface/python_arithmetic.js";
import { withGraphChange } from "./interface/region.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.VideoPlayerUI";
const NODE_NAMES = ["WASLoadVideo", "WASLoadVideoUpload"];
const SETTING_ID = "WAS.LoadVideo.ShowPlayer";

const FILE_WIDGET = "file";
const START_WIDGET = "start";
const END_WIDGET = "end";
const UI_WIDGET_NAME = "was_video_ui";
const UI_WIDGET_TYPE = "was_video_player";

// Every widget the strip reads. Each is watched, so changing one from the node redraws the
// selection without a run.
const WATCHED = [
  FILE_WIDGET, START_WIDGET, END_WIDGET, "num_frames", "strategy", "nth", "seed", "target_fps",
];

// Height of the appended widget in node units, the most it grows to, and the narrowest it is
// worth drawing in. Below that width the strip is too coarse to place a handle on.
const UI_HEIGHT = 220;
const MAX_UI_HEIGHT = 620;
const MIN_UI_WIDTH = 260;

// The control band under the picture, in element pixels, and the parts inside it.
const STRIP_HEIGHT = 46;
const PAD = 8;
const GLYPH = 12;
const TRACK_TOP = 6;
const TRACK_HEIGHT = 20;
const INFO_HEIGHT = 12;
const TIME_WIDTH = 82;
const HANDLE_WIDTH = 3;

// How near the pointer has to come to a handle, in element pixels, to take hold of it.
const GRAB = 7;

// The ceiling `modules/media/reader.py` holds one read to.
const MAX_FRAMES = 4096;

// What a clip is measured against before its own rate has been watched. Nothing on the server
// tells the page how many frames a file holds, so the count is a duration times a rate.
const FALLBACK_FPS = 30;

// The rates a measurement is drawn to when it lands near one, so a clip encoded at 29.97 is
// counted at 29.97 rather than at whatever the compositor averaged out to.
const KNOWN_RATES = [8, 10, 12, 15, 16, 20, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];
const RATE_TOLERANCE = 0.02;

// How much playback a rate is measured over before it is believed.
const RATE_SAMPLE_FRAMES = 12;
const RATE_SAMPLE_SECONDS = 0.35;

// How long the silent probe waits for a rate before giving up and leaving the fallback in place.
const PROBE_TIMEOUT_MS = 2000;

// A fresh upload is read while it is still being written, so a failed load is tried again
// before the stand-in is drawn. The wait grows with each attempt.
const LOAD_RETRIES = 3;
const LOAD_RETRY_MS = 400;

// The route answering what a clip actually measures. `requestVideoFrameCallback` counts the
// frames the compositor presented, which is fewer than the file holds whenever the browser
// cannot keep up, and a handle dragged against that count means a different frame to the node.
const MEASURE_ROUTE = "/was/interface/api/video_probe";

// When the empty wrappers are swept, in milliseconds from the panel being attached. The last
// pass also drops the watch that looks for more of them.
const SWEEP_DELAYS = [200, 1000, 3000];

/**
 * Clamp a number into a range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} low - Lower bound.
 * @param {number} high - Upper bound.
 * @returns {number} The value, held inside the bounds.
 */
function clamp(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

/**
 * Find one of a node's widgets by name.
 *
 * @param {object} node - The node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * One widget's value.
 *
 * @param {object} node - The node holding it.
 * @param {string} name - Widget name.
 * @param {*} fallback - What to answer where the node carries no such widget.
 * @returns {*} The value.
 */
function widgetValue(node, name, fallback) {
  const widget = findWidget(node, name);
  return widget ? widget.value ?? fallback : fallback;
}

/**
 * Read a widget as a whole number.
 *
 * @param {object} node - The node holding it.
 * @param {string} name - Widget name.
 * @param {number} fallback - What to answer for a missing or unreadable value.
 * @returns {number} The value, rounded.
 */
function intValue(node, name, fallback) {
  const number = Number(widgetValue(node, name, fallback));
  return Number.isFinite(number) ? Math.round(number) : fallback;
}

/**
 * The half-open range of source frames a start and an end name.
 *
 * @param {number} total - Frames in the clip.
 * @param {number} start - First frame, counting from 0. Negative counts back from the end.
 * @param {number} end - Last frame, inclusive. Negative counts back from the end, so -1 is the
 *   final frame.
 * @returns {{first: number, stop: number}} The first frame and one past the last.
 */
function sliceBounds(total, start, end) {
  // The same arithmetic as `slice_bounds` in `modules/media/sampling.py`, including its answer
  // for a range with nothing in it, so the strip lights what the node would actually read.
  if (!(total > 0)) return { first: 0, stop: 0 };
  const first = clamp(start < 0 ? start + total : start, 0, total - 1);
  const last = clamp(end < 0 ? end + total : end, 0, total - 1);
  if (last < first) return { first: 0, stop: total };
  return { first, stop: last + 1 };
}

/**
 * The value `end` takes for a half-open range.
 *
 * @param {number} stop - One past the last frame of the range.
 * @param {number} total - Frames in the clip.
 * @returns {number} The last frame's index, or -1 where the range runs to the end of the clip.
 */
function endValue(stop, total) {
  // -1 rather than the last index, so a range that runs to the end still runs to the end when
  // the clip is measured again at a rate the page had not seen when the handle was dragged.
  return stop >= total ? -1 : Math.max(0, stop - 1);
}

/**
 * Ask the node what a clip measures.
 *
 * @param {string} value - What the file widget holds.
 * @returns {Promise<object|null>} The measurement, or null where it could not be had.
 */
async function measureFile(value) {
  const text = String(value ?? "").trim();
  if (!text) return null;
  try {
    const query = new URLSearchParams({ file: text });
    const response = await fetchWithin(`${MEASURE_ROUTE}?${query}`);
    if (!response?.ok) return null;
    const found = await response.json();
    return found?.read && found.fps > 0 && found.frame_count > 0 ? found : null;
  } catch (error) {
    return null;
  }
}

/**
 * A measured rate, drawn to a standard one where it is near enough.
 *
 * @param {number} rate - Frames per second, as measured.
 * @returns {number} The nearest known rate within `RATE_TOLERANCE`, or the measurement itself.
 */
function snapRate(rate) {
  if (!(rate > 0)) return FALLBACK_FPS;
  let best = rate;
  let nearest = RATE_TOLERANCE;
  for (const known of KNOWN_RATES) {
    const offset = Math.abs(known - rate) / known;
    if (offset <= nearest) {
      nearest = offset;
      best = known;
    }
  }
  return best;
}

/**
 * A time in seconds, as the strip writes it.
 *
 * @param {number} seconds - The time.
 * @returns {string} Minutes, seconds and tenths.
 */
function formatTime(seconds) {
  const whole = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(whole / 60);
  const rest = Math.floor(whole % 60);
  const tenths = Math.floor((whole % 1) * 10);
  return `${minutes}:${String(rest).padStart(2, "0")}.${tenths}`;
}

/**
 * The address a chosen file plays from.
 *
 * @param {string} value - What the file widget holds.
 * @returns {string} The view route with its parameters, or the empty string for no file.
 */
function viewUrl(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  // A file outside the input folder is named `clip.mp4 [temp]`, which is the annotation
  // `folder_paths` reads and not part of the name the route is asked for.
  const annotated = /^(.*)\s+\[(\w+)\]$/.exec(text);
  const named = annotated ? annotated[1] : text;
  const kind = annotated ? annotated[2] : "input";
  const cut = Math.max(named.lastIndexOf("/"), named.lastIndexOf("\\"));
  const query = new URLSearchParams({
    filename: cut >= 0 ? named.slice(cut + 1) : named,
    subfolder: cut >= 0 ? named.slice(0, cut) : "",
    type: kind,
  });
  return api.apiURL(`/view?${query}`);
}

/**
 * Read whether the editor is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * The property descriptor for a name, wherever on the chain it is defined.
 *
 * @param {object} target - The object to look at.
 * @param {string} name - The property.
 * @returns {object|null} The descriptor, or null where nothing on the chain defines it.
 */
function findDescriptor(target, name) {
  let cursor = target;
  while (cursor) {
    const found = Object.getOwnPropertyDescriptor(cursor, name);
    if (found) return found;
    cursor = Object.getPrototypeOf(cursor);
  }
  return null;
}

// The root tag the file menu's labels end in, as `clip.mp4 [input]`.
const TAGGED = /\s\[(input|output|temp)\]$/;

/**
 * Give a bare file name the input tag the menu's labels carry, as the upload button stores one.
 *
 * @param {object} widget - The file widget.
 * @returns {boolean} True where the value was rewritten.
 */
function tagUpload(widget) {
  const value = widget?.value;
  const values = widget?.options?.values;
  if (typeof value !== "string" || !value || TAGGED.test(value) || !Array.isArray(values)) return false;
  if (!values.some((entry) => typeof entry === "string" && TAGGED.test(entry))) return false;
  const label = `${value} [input]`;
  const bare = values.indexOf(value);
  if (bare >= 0) values.splice(bare, 1);
  if (!values.includes(label)) values.push(label);
  widget.value = label;
  return true;
}

/**
 * Call back whenever a widget's value is written, however it was written.
 *
 * @param {object} widget - The widget to watch.
 * @param {(value: *) => void} onChange - Run after the value lands.
 * @returns {void}
 */
function watchValue(widget, onChange) {
  if (!widget || widget.__was_video_watched) return;
  widget.__was_video_watched = true;
  // The upload button, the frontend's own number controls and a workflow load all assign to
  // `value` without going through the widget's callback, so the property is what is watched and
  // the callback is left to whoever else has chained onto it.
  const descriptor = findDescriptor(widget, "value");
  if (typeof descriptor?.get === "function" && typeof descriptor?.set === "function") {
    const { get, set } = descriptor;
    Object.defineProperty(widget, "value", {
      configurable: true,
      enumerable: true,
      get() {
        return get.call(this);
      },
      set(next) {
        set.call(this, next);
        onChange(next);
      },
    });
    return;
  }
  let held = widget.value;
  Object.defineProperty(widget, "value", {
    configurable: true,
    enumerable: true,
    get() {
      return held;
    },
    set(next) {
      held = next;
      onChange(next);
    },
  });
}

/**
 * Hide the empty widget wrappers the frontend leaves around a video upload combo.
 *
 * @param {HTMLElement} element - The panel, for the wrapper it was mounted in.
 * @returns {void}
 */
function neutraliseOverlays(element) {
  const host = element?.closest?.(".dom-widget");
  const parent = host?.parentElement;
  if (!parent) return;
  for (const sibling of parent.children) {
    if (sibling === host || !sibling.classList?.contains("dom-widget")) continue;
    // A combo declared with an upload type gets a dom-widget overlay of its own, and the Vue
    // frontend leaves it holding nothing but comment placeholders. It has a box, it is laid over
    // the panel, and every pointer event that lands on it stops there.
    const filled = Array.from(sibling.childNodes)
      .some((child) => child.nodeType === Node.ELEMENT_NODE);
    if (filled) continue;
    sibling.style.pointerEvents = "none";
    sibling.style.display = "none";
  }
}

/**
 * Hide the plain preview the frontend builds for a video upload combo.
 *
 * @param {object} node - The node to sweep.
 * @param {string} keep - The name of this panel's own widget.
 * @returns {void}
 */
function hidePlainPreview(node, keep) {
  for (const widget of node?.widgets ?? []) {
    if (!widget || widget.name === keep) continue;
    const element = widget.element ?? widget.inputEl;
    if (!element) continue;
    if (element.tagName !== "VIDEO" && !element.querySelector?.("video")) continue;
    element.style.display = "none";
    // Hidden and flattened rather than spliced out of `node.widgets`: `serialize` writes
    // `widgets_values` by absolute index, so dropping one moves every later value into the wrong
    // widget the next time the workflow is loaded.
    widget.computeSize = () => [0, -4];
    widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
  }
}

/**
 * Where the picture, the track and the info line sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the bands and of the parts of the strip.
 */
function computeLayout(width, height) {
  const stripTop = Math.max(0, height - STRIP_HEIGHT);
  const trackY = stripTop + TRACK_TOP;
  const trackX = PAD + GLYPH + 8;
  const trackWidth = Math.max(0, width - trackX - PAD - TIME_WIDTH);
  return {
    width,
    height,
    stripTop,
    glyphX: PAD,
    glyphY: trackY + (TRACK_HEIGHT - GLYPH) / 2,
    trackX,
    trackY,
    trackWidth,
    infoY: trackY + TRACK_HEIGHT + 4,
  };
}

/**
 * Draw the play or the pause glyph.
 *
 * @param {CanvasRenderingContext2D} ctx - Where to draw.
 * @param {object} layout - Geometry from `computeLayout`.
 * @param {string} colour - Fill colour.
 * @param {boolean} playing - True to draw the pause bars instead of the triangle.
 * @returns {{x: number, y: number, width: number, height: number}} The area it covers.
 */
function drawPlayGlyph(ctx, layout, colour, playing) {
  const x = layout.glyphX;
  const y = layout.glyphY;
  ctx.save();
  ctx.fillStyle = colour;
  if (playing) {
    const bar = GLYPH * 0.32;
    ctx.fillRect(x, y, bar, GLYPH);
    ctx.fillRect(x + GLYPH - bar, y, bar, GLYPH);
  } else {
    ctx.beginPath();
    ctx.moveTo(x + 1, y);
    ctx.lineTo(x + GLYPH, y + GLYPH / 2);
    ctx.lineTo(x + 1, y + GLYPH);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
  return { x, y, width: GLYPH, height: GLYPH };
}

/**
 * Build the timeline editor for one node.
 *
 * @param {object} node - The node the editor belongs to.
 * @returns {object} A panel for `appendInterfaceWidget`, with the hooks the node chains onto.
 */
function createVideoPlayer(node) {
  const element = document.createElement("div");
  element.style.cssText = "width:100%;height:100%;position:relative;overflow:hidden;";

  const video = document.createElement("video");
  video.style.cssText = "position:absolute;left:0;top:0;width:100%;"
    + `height:calc(100% - ${STRIP_HEIGHT}px);object-fit:contain;display:block;background:#000;`;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.controls = false;
  video.preload = "metadata";
  video.disablePictureInPicture = true;
  video.disableRemotePlayback = true;
  video.setAttribute("controlslist", "nodownload nofullscreen noremoteplayback noplaybackrate");
  element.appendChild(video);

  // Laid over the picture as well as under it, so one surface carries every gesture.
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;display:block;"
    + "touch-action:none;";
  element.appendChild(canvas);

  const state = {
    url: "",
    duration: 0,
    fps: FALLBACK_FPS,
    rateKnown: false,
    rateWatching: false,
    // A clip nobody has played yet is counted at FALLBACK_FPS, and a handle dragged against
    // that count means a different frame to the node than the one it was dropped on. The probe
    // below plays a fraction of a second silently to measure the real rate first.
    rateProbed: false,
    retries: 0,
    // Frames the node counted, which beats deriving them from the duration and the rate.
    measuredFrames: 0,
    failed: false,
    // Set by a click on the picture or on the glyph. A pinned clip keeps playing once the
    // pointer has left, and an unpinned one is only previewed while the pointer is over it.
    pinned: false,
    dragging: null,
    dragFrom: 0,
    dragFirst: 0,
    dragStop: 0,
    // The range a live drag stands at, as `{first, stop}`, before it is put on the widgets. It
    // is null between gestures, and everything that reads the range reads it while it is set.
    pending: null,
    hover: null,
    disposed: false,
    painting: false,
    ticking: false,
    writing: false,
  };

  const hover = hoverTitles(element);
  const timers = [];
  let releaseRatio = null;
  let observer = null;
  // What the last frame count was worked out from, so a repaint during playback does not draw a
  // fresh pick out of a clip thousands of frames long on every frame.
  let keptMemo = { key: "", count: 0 };

  /**
   * How many frames the clip holds, at the rate it is currently measured against.
   *
   * @returns {number} The count, or 0 while nothing has loaded.
   */
  function frameTotal() {
    if (state.measuredFrames > 0) return state.measuredFrames;
    if (!(state.duration > 0) || !(state.fps > 0)) return 0;
    return Math.max(1, Math.round(state.duration * state.fps));
  }

  /**
   * The range on show: a live drag's, or the one the two widgets name.
   *
   * @returns {{first: number, stop: number}} The first frame and one past the last.
   */
  function readRange() {
    if (state.pending) return { ...state.pending };
    const start = intValue(node, START_WIDGET, 0);
    const end = intValue(node, END_WIDGET, -1);
    return sliceBounds(frameTotal(), start, end);
  }

  /**
   * How many frames the node would answer for a range.
   *
   * @param {number} frames - Frames in the range.
   * @returns {number} The count, after the rate change, the strategy and the read ceiling.
   */
  function keptCount(frames) {
    if (!(frames > 0)) return 0;
    const target = Number(widgetValue(node, "target_fps", 0)) || 0;
    const wanted = Math.max(0, intValue(node, "num_frames", 0));
    const strategy = String(widgetValue(node, "strategy", "uniform"));
    const nth = Math.max(1, intValue(node, "nth", 2));
    const seed = Number(widgetValue(node, "seed", 0)) || 0;
    const key = `${frames}|${state.fps}|${target}|${wanted}|${strategy}|${nth}|${seed}`;
    if (key === keptMemo.key) return keptMemo.count;

    // The rate change comes first and the strategy picks out of what it leaves, which is the
    // order `reader.read` applies them in, down to how it rounds a count landing on a half.
    const window = target > 0 && state.fps > 0
      ? Math.max(1, roundHalfEven((frames * target) / state.fps))
      : frames;
    const picked = wanted ? selection(window, wanted, strategy, nth, seed).length : window;
    const count = Math.min(picked || Math.min(window, wanted), MAX_FRAMES);
    keptMemo = { key, count };
    return count;
  }

  /**
   * Put a range back on the two widgets, as one undo step.
   *
   * @param {number} first - First frame of the range.
   * @param {number} stop - One past its last frame.
   * @returns {void}
   */
  function commit(first, stop) {
    const startWidget = findWidget(node, START_WIDGET);
    const endWidget = findWidget(node, END_WIDGET);
    const total = frameTotal();
    if (!startWidget || !endWidget || !(total > 0)) return;

    const held = clamp(first, 0, total - 1);
    const past = clamp(stop, held + 1, total);
    // The end the gesture left alone keeps the number it was typed with, so a start of -60 is
    // still sixty frames from the end after the other handle is dragged.
    let nextStart = intValue(node, START_WIDGET, 0);
    let nextEnd = intValue(node, END_WIDGET, -1);
    const standing = sliceBounds(total, nextStart, nextEnd);
    if (standing.first !== held) nextStart = held;
    if (standing.stop !== past) nextEnd = endValue(past, total);
    // Both written outright where the kept spelling no longer names this range, which is what a
    // pair holding an end below the start resolves to.
    const reached = sliceBounds(total, nextStart, nextEnd);
    if (reached.first !== held || reached.stop !== past) {
      nextStart = held;
      nextEnd = endValue(past, total);
    }
    if (startWidget.value === nextStart && endWidget.value === nextEnd) return;

    state.writing = true;
    try {
      withGraphChange(() => {
        if (startWidget.value !== nextStart) {
          startWidget.value = nextStart;
          startWidget.callback?.(nextStart);
        }
        if (endWidget.value !== nextEnd) {
          endWidget.value = nextEnd;
          endWidget.callback?.(nextEnd);
        }
      });
    } finally {
      state.writing = false;
    }
    keptMemo = { key: "", count: 0 };
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Finish a drag, with or without the range it stood at.
   *
   * @param {boolean} keep - True to put the range on the widgets, false to drop it.
   * @param {number} [pointerId] - The pointer the gesture was captured on.
   * @returns {void}
   */
  function endDrag(keep, pointerId) {
    if (!state.dragging) return;
    state.dragging = null;
    if (pointerId !== undefined && element.hasPointerCapture?.(pointerId)) {
      element.releasePointerCapture?.(pointerId);
    }
    const range = state.pending;
    state.pending = null;
    // One write for the whole gesture. Written on every move instead, the change tracker takes a
    // snapshot per move and one undo step gives back a fraction of the drag.
    if (keep && range) commit(range.first, range.stop);
    holdInsideRange();
    schedulePaint();
  }

  /**
   * Repaint on the next frame, once however many changes arrive this one.
   *
   * @returns {void}
   */
  function schedulePaint() {
    if (state.painting || state.disposed) return;
    state.painting = true;
    requestAnimationFrame(() => {
      state.painting = false;
      if (!state.disposed) paint();
    });
  }

  /**
   * Repaint every frame for as long as the clip is playing.
   *
   * @returns {void}
   */
  function runTicker() {
    if (state.ticking || state.disposed) return;
    state.ticking = true;
    const step = () => {
      if (state.disposed || video.paused) {
        state.ticking = false;
        schedulePaint();
        return;
      }
      holdInsideRange();
      paint();
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /**
   * Keep a playing clip inside the range the handles mark.
   *
   * @returns {void}
   */
  function holdInsideRange() {
    const total = frameTotal();
    if (video.paused || !(state.duration > 0) || !(total > 0)) return;
    const { first, stop } = readRange();
    const from = (first / total) * state.duration;
    const to = (stop / total) * state.duration;
    if (!(to > from)) return;
    // One frame of slack under the start, since a seek lands on the nearest keyframe it can
    // decode from and may come back a little short of where it was sent.
    const frame = state.duration / total;
    if (video.currentTime >= to || video.currentTime < from - frame) video.currentTime = from;
  }

  /**
   * Measure the clip's rate from the frames the compositor presents.
   *
   * @returns {void}
   */
  function watchRate() {
    if (state.rateKnown || state.rateWatching) return;
    if (typeof video.requestVideoFrameCallback !== "function") return;
    state.rateWatching = true;
    let firstTime = null;
    let firstFrames = 0;
    const step = (now, metadata) => {
      if (state.disposed) return;
      const at = Number(metadata?.mediaTime);
      const frames = Number(metadata?.presentedFrames);
      if (Number.isFinite(at) && Number.isFinite(frames)) {
        // A loop back to the start ends the window rather than making it look enormous.
        if (firstTime === null || at < firstTime) {
          firstTime = at;
          firstFrames = frames;
        } else {
          const span = at - firstTime;
          const seen = frames - firstFrames;
          if (span >= RATE_SAMPLE_SECONDS && seen >= RATE_SAMPLE_FRAMES) {
            state.fps = snapRate(seen / span);
            state.rateKnown = true;
            state.rateWatching = false;
            keptMemo = { key: "", count: 0 };
            schedulePaint();
            return;
          }
        }
      }
      video.requestVideoFrameCallback(step);
    };
    video.requestVideoFrameCallback(step);
  }

  /**
   * Point the player at a file.
   *
   * @param {string} value - What the file widget holds.
   * @returns {void}
   */
  function loadFile(value) {
    const url = viewUrl(value);
    // A failed element is asked again even when the address has not moved. The upload button
    // sets the widget to the name it is still writing, so the first load reads a part-written
    // file and the element latches an error; without this the same choice never retries and
    // the node shows the stand-in for good.
    if (url === state.url && !state.failed) return;
    state.url = url;
    state.retries = 0;
    state.duration = 0;
    state.fps = FALLBACK_FPS;
    state.rateKnown = false;
    state.rateWatching = false;
    state.rateProbed = false;
    state.measuredFrames = 0;
    state.failed = false;
    state.pinned = false;
    keptMemo = { key: "", count: 0 };
    video.pause();
    if (url) video.src = url;
    else video.removeAttribute("src");
    // Asked for again either way, so the decoder lets go of whatever was there before.
    video.load();
    schedulePaint();
  }

  /**
   * Move the head to the start of the range when it is outside it.
   *
   * @returns {void}
   */
  function enterRange() {
    const total = frameTotal();
    if (!(state.duration > 0) || !(total > 0)) return;
    const { first, stop } = readRange();
    const from = (first / total) * state.duration;
    const to = (stop / total) * state.duration;
    if (video.currentTime < from || video.currentTime >= to) video.currentTime = from;
  }

  /**
   * Start or stop playback, and pin it either way.
   *
   * @returns {void}
   */
  function togglePlay() {
    if (!(state.duration > 0)) return;
    if (!video.paused) {
      state.pinned = false;
      video.pause();
      return;
    }
    state.pinned = true;
    enterRange();
    // A click is the gesture the browser wants before it will play sound, so a pinned clip is
    // audible and a clip previewed under the pointer is not. A refusal is answered by playing
    // it muted, which is what a browser holding a stricter policy will take.
    video.muted = false;
    video.play().catch(() => {
      video.muted = true;
      video.play().catch(() => {});
    });
  }

  /**
   * Draw the whole editor.
   *
   * @returns {void}
   */
  function paint() {
    const ratio = surfaceRatio(element);
    const width = Math.max(1, Math.round(element.clientWidth));
    const height = Math.max(1, Math.round(element.clientHeight));
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    // Cleared rather than filled: the picture is behind the canvas, so the band over it has to
    // stay transparent.
    ctx.clearRect(0, 0, width, height);

    const theme = readTheme();
    const layout = computeLayout(width, height);
    const titles = [];
    drawPicture(ctx, layout, theme, titles);
    drawStrip(ctx, layout, theme, titles);
    hover.set(titles);
  }

  /**
   * Draw the stand-in where no clip has loaded, and make the picture a play button.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {Array<object>} titles - Hover regions, appended to.
   * @returns {void}
   */
  function drawPicture(ctx, layout, theme, titles) {
    if (!(layout.stripTop > 0)) return;
    if (!(state.duration > 0)) {
      ctx.fillStyle = theme.bgDark;
      ctx.fillRect(0, 0, layout.width, layout.stripTop);
      drawStandIn(ctx, { x: 0, y: 0, w: layout.width, h: layout.stripTop });
      return;
    }
    titles.push({
      x: 0,
      y: 0,
      width: layout.width,
      height: layout.stripTop,
      title: video.paused ? "Play clip" : "Pause clip",
    });
  }

  /**
   * Draw the control band: the glyph, the track, the handles and the line under them.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {Array<object>} titles - Hover regions, appended to.
   * @returns {void}
   */
  function drawStrip(ctx, layout, theme, titles) {
    ctx.fillStyle = theme.panelBg;
    ctx.fillRect(0, layout.stripTop, layout.width, STRIP_HEIGHT);

    const glyph = drawPlayGlyph(ctx, layout, theme.fgMuted, !video.paused);
    titles.push({ ...glyph, title: video.paused ? "Play clip" : "Pause clip" });

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.trackX, layout.trackY, layout.trackWidth, TRACK_HEIGHT);
    ctx.strokeStyle = theme.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(
      Math.round(layout.trackX) + 0.5,
      Math.round(layout.trackY) + 0.5,
      Math.max(0, Math.round(layout.trackWidth) - 1),
      TRACK_HEIGHT - 1,
    );

    const total = frameTotal();
    if (!(total > 0) || !(layout.trackWidth > 0)) {
      const why = state.failed ? "clip could not load" : "no clip chosen";
      drawInfo(ctx, layout, theme, titles, why, "");
      return;
    }

    const { first, stop } = readRange();
    const from = layout.trackX + (first / total) * layout.trackWidth;
    const to = layout.trackX + (stop / total) * layout.trackWidth;
    const rightmost = layout.trackX + layout.trackWidth - HANDLE_WIDTH;

    if (state.duration > 0) {
      const played = clamp(video.currentTime / state.duration, 0, 1);
      ctx.fillStyle = theme.borderLight;
      ctx.fillRect(layout.trackX, layout.trackY, layout.trackWidth * played, TRACK_HEIGHT);
    }

    ctx.fillStyle = theme.accentBg;
    ctx.fillRect(from, layout.trackY, Math.max(1, to - from), TRACK_HEIGHT);

    const grabbed = state.dragging ?? state.hover;
    for (const [edge, at] of [["start", from], ["end", to - HANDLE_WIDTH]]) {
      ctx.fillStyle = grabbed === edge ? theme.accentHover : theme.accent;
      ctx.fillRect(clamp(at, layout.trackX, rightmost), layout.trackY, HANDLE_WIDTH, TRACK_HEIGHT);
    }

    if (state.duration > 0) {
      const played = clamp(video.currentTime / state.duration, 0, 1);
      const head = layout.trackX + played * layout.trackWidth;
      ctx.fillStyle = theme.fg;
      ctx.fillRect(clamp(head - 1, layout.trackX, rightmost), layout.trackY, 2, TRACK_HEIGHT);
    }

    // The handles come before the track: the first region holding a position is the one whose
    // text is shown, and the track covers both of them.
    titles.push({
      x: from - GRAB,
      y: layout.trackY,
      width: GRAB * 2,
      height: TRACK_HEIGHT,
      title: "Drag to set start",
    });
    titles.push({
      x: to - GRAB,
      y: layout.trackY,
      width: GRAB * 2,
      height: TRACK_HEIGHT,
      title: "Drag to set end",
    });
    titles.push({
      x: layout.trackX,
      y: layout.trackY,
      width: layout.trackWidth,
      height: TRACK_HEIGHT,
      title: "Click to seek",
    });

    ctx.font = "10px sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ctx.fillStyle = theme.fgMuted;
    const clock = `${formatTime(video.currentTime)} / ${formatTime(state.duration)}`;
    ctx.fillText(clock, layout.width - PAD, layout.trackY + TRACK_HEIGHT / 2);

    const range = `${first} to ${stop - 1} of ${total - 1}`;
    drawInfo(ctx, layout, theme, titles, range, `${keptCount(stop - first)} frames`);
  }

  /**
   * Draw the line under the track.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {Array<object>} titles - Hover regions, appended to.
   * @param {string} left - What the range says.
   * @param {string} right - What comes out of it, empty for nothing to say.
   * @returns {void}
   */
  function drawInfo(ctx, layout, theme, titles, left, right) {
    const middle = layout.infoY + INFO_HEIGHT / 2;
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(left, PAD, middle);
    if (!right) return;

    ctx.textAlign = "right";
    ctx.fillText(right, layout.width - PAD, middle);
    // The count is a duration times a rate rather than a number read out of the container, so it
    // is drawn as an estimate however close the measurement has come.
    const box = drawIcon(
      ctx,
      ICON.APPROXIMATE,
      layout.width - PAD - ctx.measureText(right).width - 4 - ICON_SIZE,
      middle - ICON_SIZE / 2,
      ICON_SIZE,
      theme.fgMuted,
    );
    const detail = state.rateKnown ? "rate measured on playback" : "rate not measured yet";
    titles.push({ ...box, title: iconTitle(ICON.APPROXIMATE, detail) });
  }

  /**
   * Which part of the editor a position is over.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {{x: number, y: number}} at - Pointer position in element pixels.
   * @returns {string|null} One of `picture`, `glyph`, `start`, `end`, `body`, `track`, or null.
   */
  function partAt(layout, at) {
    if (at.y < layout.stripTop) return "picture";
    if (
      at.x >= layout.glyphX - 3 && at.x <= layout.glyphX + GLYPH + 3 &&
      at.y >= layout.glyphY - 4 && at.y <= layout.glyphY + GLYPH + 4
    ) {
      return "glyph";
    }
    const total = frameTotal();
    if (!(total > 0) || !(layout.trackWidth > 0)) return null;
    if (at.y < layout.trackY || at.y > layout.trackY + TRACK_HEIGHT) return null;
    if (at.x < layout.trackX - GRAB || at.x > layout.trackX + layout.trackWidth + GRAB) return null;

    const { first, stop } = readRange();
    const from = layout.trackX + (first / total) * layout.trackWidth;
    const to = layout.trackX + (stop / total) * layout.trackWidth;
    const nearStart = Math.abs(at.x - from);
    const nearEnd = Math.abs(at.x - to);
    // The handles take the gesture before the band between them does, so a short selection can
    // still be widened rather than only moved.
    if (nearStart <= GRAB || nearEnd <= GRAB) return nearStart <= nearEnd ? "start" : "end";
    if (at.x > from && at.x < to) return "body";
    return "track";
  }

  /**
   * Move the head to a position on the track.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {{x: number, y: number}} at - Pointer position in element pixels.
   * @returns {void}
   */
  function seek(layout, at) {
    if (!(state.duration > 0) || !(layout.trackWidth > 0)) return;
    video.currentTime = clamp((at.x - layout.trackX) / layout.trackWidth, 0, 1) * state.duration;
    schedulePaint();
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;

    const layout = computeLayout(element.clientWidth, element.clientHeight);
    const at = elementPoint(element, event);
    const part = partAt(layout, at);
    if (part === "picture" || part === "glyph") {
      togglePlay();
      schedulePaint();
      event.stopPropagation();
      event.preventDefault();
      return;
    }
    if (part === "track") {
      seek(layout, at);
      event.stopPropagation();
      event.preventDefault();
      return;
    }
    if (part !== "start" && part !== "end" && part !== "body") return;

    const range = readRange();
    state.dragging = part;
    state.dragFrom = at.x;
    state.dragFirst = range.first;
    state.dragStop = range.stop;
    state.pending = null;
    element.setPointerCapture?.(event.pointerId);
    event.stopPropagation();
    event.preventDefault();
    schedulePaint();
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    const layout = computeLayout(element.clientWidth, element.clientHeight);
    const at = elementPoint(element, event);
    if (!state.dragging) {
      const part = partAt(layout, at);
      const held = part === "start" || part === "end" ? part : null;
      if (held !== state.hover) {
        state.hover = held;
        schedulePaint();
      }
      return;
    }
    // A button released over another window, or a capture the browser took away, ends the
    // gesture without a pointerup. Without this the handle would keep following an unpressed
    // pointer and commit a range nobody chose.
    if (!(event.buttons & 1)) {
      endDrag(false, event.pointerId);
      return;
    }

    const total = frameTotal();
    if (!(total > 0) || !(layout.trackWidth > 0)) return;
    const moved = Math.round(((at.x - state.dragFrom) / layout.trackWidth) * total);
    if (state.dragging === "start") {
      const first = clamp(state.dragFirst + moved, 0, state.dragStop - 1);
      state.pending = { first, stop: state.dragStop };
    } else if (state.dragging === "end") {
      const stop = clamp(state.dragStop + moved, state.dragFirst + 1, total);
      state.pending = { first: state.dragFirst, stop };
    } else {
      const span = state.dragStop - state.dragFirst;
      const first = clamp(state.dragFirst + moved, 0, Math.max(0, total - span));
      state.pending = { first, stop: first + span };
    }
    event.preventDefault();
    schedulePaint();
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    endDrag(true, event.pointerId);
  };

  const onDragLost = (event) => endDrag(false, event.pointerId);

  const onPointerEnter = () => {
    // One clip on the page plays at a time this way, which keeps a graph full of loaders from
    // holding open a stream per node.
    if (state.pinned || !(state.duration > 0) || !video.paused) return;
    video.muted = true;
    enterRange();
    video.play().catch(() => {});
  };

  const onPointerLeave = () => {
    if (state.hover) {
      state.hover = null;
      schedulePaint();
    }
    if (state.pinned || video.paused) return;
    video.pause();
  };

  const onDoubleClick = (event) => {
    const layout = computeLayout(element.clientWidth, element.clientHeight);
    const part = partAt(layout, elementPoint(element, event));
    if (part !== "start" && part !== "end" && part !== "body") return;
    state.dragging = null;
    state.pending = null;
    commit(0, frameTotal());
    schedulePaint();
    event.preventDefault();
  };

  const onContextMenu = (event) => {
    // The browser's own menu over a video offers to download it and to loop it, neither of
    // which is this node's, and the canvas behind the panel never sees the event either way.
    event.preventDefault();
  };

  /**
   * Measure the clip's real frame rate before anything is dragged against it.
   *
   * @returns {void}
   */
  function probeRate() {
    if (state.rateProbed || state.rateKnown || state.disposed) return;
    state.rateProbed = true;

    // The node's own measurement first, which is exact and costs one request.
    measureFile(widgetValue(node, FILE_WIDGET, "")).then((found) => {
      if (!found || state.rateKnown || state.disposed) return;
      state.fps = found.fps;
      state.measuredFrames = found.frame_count;
      state.rateKnown = true;
      keptMemo = { key: "", count: 0 };
      schedulePaint();
    });

    if (typeof video.requestVideoFrameCallback !== "function") return;
    if (!video.paused || !(state.duration > 0)) return;

    const wasMuted = video.muted;
    video.muted = true;
    watchRate();
    const finish = () => {
      // The user pressing play during the probe owns the clip from then on.
      if (state.pinned) return;
      try {
        video.pause();
        video.currentTime = 0;
      } catch (error) {
        console.warn(`[${EXT_NAME}] Could not rewind after measuring the rate:`, error);
      }
      video.muted = wasMuted;
      schedulePaint();
    };
    const settle = () => {
      if (state.rateKnown || state.disposed) finish();
      else if (Date.now() - began < PROBE_TIMEOUT_MS) setTimeout(settle, 60);
      else finish();
    };
    const began = Date.now();
    const started = video.play();
    if (started?.catch) {
      // Autoplay refused. Nothing is lost: the rate is measured the moment the clip is played.
      started.catch(() => { video.muted = wasMuted; state.rateProbed = false; });
    }
    setTimeout(settle, 60);
  }

  const onMetadata = () => {
    state.duration = Number(video.duration) || 0;
    state.failed = false;
    keptMemo = { key: "", count: 0 };
    schedulePaint();
    probeRate();
  };

  const onPlaying = () => {
    watchRate();
    runTicker();
  };

  const onPause = () => schedulePaint();

  const onError = () => {
    // A file still being written answers a truncated body, so the first read of a fresh
    // upload fails on a clip the browser can play perfectly a moment later.
    if (state.url && (state.retries || 0) < LOAD_RETRIES) {
      state.retries = (state.retries || 0) + 1;
      setTimeout(() => {
        if (state.disposed || state.url !== video.getAttribute("src")) return;
        video.load();
      }, LOAD_RETRY_MS * state.retries);
      return;
    }
    state.failed = true;
    state.duration = 0;
    console.warn(`[${EXT_NAME}] The chosen video could not be played:`, state.url);
    schedulePaint();
  };

  element.addEventListener("pointerdown", onPointerDown);
  element.addEventListener("pointermove", onPointerMove);
  element.addEventListener("pointerup", onPointerUp);
  element.addEventListener("pointercancel", onDragLost);
  element.addEventListener("lostpointercapture", onDragLost);
  element.addEventListener("pointerenter", onPointerEnter);
  element.addEventListener("pointerleave", onPointerLeave);
  element.addEventListener("dblclick", onDoubleClick);
  element.addEventListener("contextmenu", onContextMenu);
  // The player scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(element);
  video.addEventListener("loadedmetadata", onMetadata);
  video.addEventListener("durationchange", onMetadata);
  video.addEventListener("playing", onPlaying);
  video.addEventListener("pause", onPause);
  video.addEventListener("error", onError);

  try {
    releaseRatio = watchSurfaceRatio(element, schedulePaint);
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to watch the drawing resolution:`, error);
  }
  // The player is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let releaseTheme = onThemeChange(schedulePaint);
  loadPlaceholder().then(() => schedulePaint());

  const last = SWEEP_DELAYS[SWEEP_DELAYS.length - 1];
  for (const delay of SWEEP_DELAYS) {
    timers.push(setTimeout(() => {
      if (state.disposed) return;
      neutraliseOverlays(element);
      hidePlainPreview(node, UI_WIDGET_NAME);
      if (delay !== last) return;
      // The wrappers are built while the node is first drawn, so the watch is dropped once that
      // is over rather than left running for the life of the page.
      observer?.disconnect();
      observer = null;
    }, delay));
  }
  timers.push(setTimeout(() => {
    if (state.disposed) return;
    const parent = element.closest?.(".dom-widget")?.parentElement;
    if (!parent) return;
    let pending = false;
    observer = new MutationObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (!state.disposed) neutraliseOverlays(element);
      });
    });
    observer.observe(parent, { childList: true });
  }, SWEEP_DELAYS[0]));

  return {
    element,
    height: UI_HEIGHT,
    maxHeight: MAX_UI_HEIGHT,
    minWidth: MIN_UI_WIDTH,
    schedulePaint,
    handleFileChanged() {
      loadFile(widgetValue(node, FILE_WIDGET, ""));
    },
    handleRangeChanged() {
      if (state.writing) return;
      keptMemo = { key: "", count: 0 };
      schedulePaint();
    },
    dispose() {
      if (state.disposed) return;
      state.disposed = true;
      for (const timer of timers) clearTimeout(timer);
      timers.length = 0;
      element.removeEventListener("pointerdown", onPointerDown);
      element.removeEventListener("pointermove", onPointerMove);
      element.removeEventListener("pointerup", onPointerUp);
      element.removeEventListener("pointercancel", onDragLost);
      element.removeEventListener("lostpointercapture", onDragLost);
      element.removeEventListener("pointerenter", onPointerEnter);
      element.removeEventListener("pointerleave", onPointerLeave);
      element.removeEventListener("dblclick", onDoubleClick);
      element.removeEventListener("contextmenu", onContextMenu);
      releaseWheel();
      video.removeEventListener("loadedmetadata", onMetadata);
      video.removeEventListener("durationchange", onMetadata);
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("error", onError);
      try {
        hover.dispose();
        releaseRatio?.();
        releaseTheme?.();
        observer?.disconnect();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to release the player's watches:`, error);
      }
      releaseRatio = null;
      releaseTheme = null;
      observer = null;
      video.pause();
      video.removeAttribute("src");
      // Asked for again with no source, which is what makes the decoder give its buffers back.
      video.load();
    },
  };
}

/**
 * Append the editor to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachVideoPlayer(node) {
  if (!findWidget(node, FILE_WIDGET) || !findWidget(node, START_WIDGET)) return;

  const player = createVideoPlayer(node);
  appendInterfaceWidget(node, player, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  for (const name of WATCHED) {
    const widget = findWidget(node, name);
    watchValue(widget, () => {
      if (name !== FILE_WIDGET) {
        player.handleRangeChanged();
        return;
      }
      // A rewritten value lands here again, already tagged.
      if (tagUpload(widget)) return;
      player.handleFileChanged();
    });
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      player.handleFileChanged();
      player.handleRangeChanged();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to reload the player after a workflow load:`, error);
    }
    return result;
  };

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      player.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the player:`, error);
    }
    return result;
  };

  player.handleFileChanged();
  player.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Load Video", "Timeline editor"],
      name: "Show the video timeline",
      tooltip:
        "Play the chosen video on Load Video and Load Video (Upload), and drag the two "
        + "handles over the strip to set start and end. Those widgets are always available on "
        + "their own. This applies to nodes added after the setting changes, so a reload shows "
        + "it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    if (proto.__was_video_wrapped) return;
    proto.__was_video_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachVideoPlayer(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the video player:`, error);
      }
      return result;
    };
  },
});
