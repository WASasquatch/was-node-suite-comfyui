/**
 * Paste region for the Image Paste Crop by Location node.
 *
 * Draws where `crop_image` lands and writes the node's `top`, `left`, `right` and `bottom`
 * widgets. One frame unit is one pixel of the image the node received.
 */

import { app } from "../../scripts/app.js";
import { imageBackdrop } from "./interface/backdrop.js";
import { EDGE, createRegionEditor } from "./interface/region.js";
import { onNodeDone } from "./interface/run_events.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.PasteCropByLocationUI";
const NODE_NAME = "Image Paste Crop by Location";
const SETTING_ID = "WAS.PasteCropByLocation.ShowInterface";

const UI_WIDGET_NAME = "was_paste_crop_by_location_ui";
const UI_WIDGET_TYPE = "was_paste_crop_by_location_region";

const TOP = "top";
const LEFT = "left";
const RIGHT = "right";
const BOTTOM = "bottom";
const CROP_BLENDING = "crop_blending";

// Each edge of the rectangle is one whole widget of this node, so the map the editor writes
// through is one to one and a gesture on one edge leaves the other three alone.
const EDGE_WIDGETS = {
  [EDGE.LEFT]: LEFT,
  [EDGE.TOP]: TOP,
  [EDGE.RIGHT]: RIGHT,
  [EDGE.BOTTOM]: BOTTOM,
};

const EDGE_NAMES = Object.values(EDGE_WIDGETS);

// The schema's own defaults, read only when a widget cannot be.
const DEFAULTS = {
  [TOP]: 0,
  [LEFT]: 0,
  [RIGHT]: 256,
  [BOTTOM]: 256,
  [CROP_BLENDING]: 0.25,
};

// Height of the appended widget in node units. The picture is drawn inside it, so it is taller
// than an interface that only draws a plot.
const UI_HEIGHT = 220;

// The node stops with an error on a rectangle with no area once it has been clamped into the
// image, so no gesture may produce one.
const MIN_SIZE = 1;

// What the rectangle and the numbers beside it are, on the footer's hover text. It holds for
// every edge and every blending, so it is reachable from the footer rather than drawn on it.
const PASTE_HOVER =
  "the four numbers are pixels of the image, and the rectangle is where crop_image lands."
  + " The soft edge is taken out of the paste rather than added around it, so a mask that never"
  + " reaches white makes the whole paste faint and a lower crop_blending brings it back";

// How many box passes `ImageFilter.GaussianBlur` runs, and how many samples one profile of the
// mask is measured at. A rectangle larger than that is measured at a reduced sampling, which
// leaves the peak where it was: the peak is set by the blur's width against the opaque core's,
// and reducing both together holds that ratio.
const BLUR_PASSES = 3;
const MAX_SAMPLES = 4096;

// A mask this close to white is white to a paste. Two blurs of a rectangle wide enough to hold
// them reach exactly one in the middle, and the rounding of the profile is what stands between.
const WHITE = 0.995;

/**
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * Test whether one of a node's inputs is linked.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  for (const input of inputs) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

/**
 * Read one of the node's numbers.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {number} fallback - What to answer when the widget cannot be read.
 * @returns {number} The widget's number, or the fallback.
 */
function readNumber(node, name, fallback) {
  const widget = findWidget(node, name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
}

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
 * Read the four widgets as a rectangle.
 *
 * @param {object} node - Node holding the widgets.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle in image pixels.
 */
function readRect(node) {
  const left = readNumber(node, LEFT, DEFAULTS[LEFT]);
  const top = readNumber(node, TOP, DEFAULTS[TOP]);
  const right = readNumber(node, RIGHT, DEFAULTS[RIGHT]);
  const bottom = readNumber(node, BOTTOM, DEFAULTS[BOTTOM]);
  return { x: left, y: top, w: right - left, h: bottom - top };
}

/**
 * Read the range one widget declares.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {{min: number, max: number}} The bounds the schema declared, wide open where they
 *   cannot be read.
 */
function widgetLimits(node, name) {
  const options = findWidget(node, name)?.options ?? {};
  const low = Number(options.min);
  const high = Number(options.max);
  return {
    min: Number.isFinite(low) ? low : Number.MIN_SAFE_INTEGER,
    max: Number.isFinite(high) ? high : Number.MAX_SAFE_INTEGER,
  };
}

/**
 * Store one edge, unless a link has taken it over.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {number} value - Position of the edge in image pixels.
 * @returns {void}
 */
function writeEdge(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget) return;
  // A widget whose input is linked is never written. The gestures refuse it before they reach
  // here, and this catches the one that cannot: a gesture still in hand while the link is
  // attached, since attaching one changes no widget value and drops nothing.
  if (inputLinked(node, name)) return;
  if (!Number.isFinite(value)) return;
  // The four inputs are declared as integers, so a whole pixel is the only thing they can
  // usefully hold, and each is held to the range its own schema states. A number past that
  // range reaches the backend as an input the prompt is refused for, naming a widget the user
  // never touched.
  const limits = widgetLimits(node, name);
  const whole = clamp(Math.round(value), limits.min, limits.max);
  if (whole === widget.value) return;
  // The write is not bracketed here. `createRegionEditor` brackets the whole gesture in the
  // canvas change events the graph's change tracker takes its snapshot from, so bracketing
  // each edge as well would split one gesture into four undo entries.
  widget.value = whole;
}

/**
 * Store the edges a gesture moved.
 *
 * @param {object} node - Node holding the widgets.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle in image
 *   pixels.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @returns {void}
 */
function writeRect(node, rect, moved) {
  const edges = {
    [EDGE.LEFT]: rect.x,
    [EDGE.TOP]: rect.y,
    [EDGE.RIGHT]: rect.x + rect.w,
    [EDGE.BOTTOM]: rect.y + rect.h,
  };
  for (const [edge, name] of Object.entries(EDGE_WIDGETS)) {
    if (moved?.[edge]) writeEdge(node, name, edges[edge]);
  }
}

/**
 * Read which edges a link has taken over.
 *
 * @param {object} node - Node to search.
 * @returns {object} A map from edge to input name, holding only the edges that are locked.
 */
function readLocks(node) {
  const locks = {};
  for (const [edge, name] of Object.entries(EDGE_WIDGETS)) {
    if (inputLinked(node, name)) locks[edge] = name;
  }
  return locks;
}

/**
 * The rectangle the node will actually paste into.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - The rectangle on screen, in image
 *   pixels.
 * @param {{width: number, height: number}} frame - The image's size in pixels, zero on either
 *   axis while it is not known.
 * @returns {{width: number, height: number}} The rectangle's size in pixels, which is zero or
 *   less on an axis whose edges cross.
 */
function pasteSize(rect, frame) {
  // The clamp is applied only once the published picture has said how big the image is. Before
  // that the widgets are the best available account of the rectangle.
  if (!(frame.width > 0) || !(frame.height > 0)) {
    return { width: rect.w, height: rect.h };
  }
  const left = clamp(rect.x, 0, frame.width);
  const right = clamp(rect.x + rect.w, 0, frame.width);
  const top = clamp(rect.y, 0, frame.height);
  const bottom = clamp(rect.y + rect.h, 0, frame.height);
  return { width: right - left, height: bottom - top };
}

/**
 * The blur radius the node feathers with.
 *
 * @param {number} width - Width of the rectangle in pixels.
 * @param {number} height - Height of the rectangle in pixels.
 * @param {number} blending - The `crop_blending` widget, held to 0 through 1 as the node holds
 *   it.
 * @returns {number} The radius `blend_ratio` carries.
 */
function blendRadius(width, height, blending) {
  // `blend_ratio` in the node is `max(crop_size) / 2 * crop_blending`.
  return (Math.max(width, height) / 2) * clamp(blending, 0, 1);
}

/**
 * The border thickness the node asks for, and the thickest one the rectangle has room for.
 *
 * @param {number} width - Width of the rectangle in pixels.
 * @param {number} height - Height of the rectangle in pixels.
 * @param {number} radius - The radius from `blendRadius`.
 * @returns {{asked: number, room: number, inset: number}} The border asked for, the room there
 *   is for one, and the thickness the node draws.
 */
function featherInset(width, height, radius) {
  const asked = Math.trunc(radius / 2);
  // The blur radius is set by the longer side, while the room for the border is set by the
  // shorter one: the node caps it at `(min(crop_size) - 1) // 2`.
  const room = Math.floor((Math.min(width, height) - 1) / 2);
  // The cap standing below the request means the border was trimmed to fit rather than fitted.
  return { asked, room, inset: Math.max(0, Math.min(asked, room)) };
}

/**
 * The radius of each box pass `ImageFilter.GaussianBlur` runs.
 *
 * @param {number} radius - The standard deviation the blur was asked for.
 * @returns {number} The radius of one pass, whole part and fraction together.
 */
function gaussianBoxRadius(radius) {
  const variance = (radius * radius) / BLUR_PASSES;
  const width = Math.sqrt(12 * variance + 1);
  const whole = Math.floor((width - 1) / 2);
  const fraction =
    ((2 * whole + 1) * (whole * (whole + 1) - 3 * variance)) /
    (6 * (variance - (whole + 1) * (whole + 1)));
  const answer = whole + fraction;
  return Number.isFinite(answer) && answer > 0 ? answer : radius;
}

/**
 * One pass of Pillow's box blur along a profile.
 *
 * @param {Float64Array} profile - Values to blur.
 * @param {number} radius - Radius of the box, whole part and fraction together.
 * @returns {Float64Array} The blurred profile.
 */
function boxPass(profile, radius) {
  const length = profile.length;
  if (!(radius > 0) || length === 0) return profile.slice();

  const out = new Float64Array(length);
  const whole = Math.trunc(radius);
  const weight = 1 / (radius * 2 + 1);
  const edge = (radius - whole) * weight;

  const sums = new Float64Array(length + 1);
  for (let index = 0; index < length; index++) sums[index + 1] = sums[index] + profile[index];
  // Pillow carries the profile's own end outward past it, so a profile padded with zeros is
  // padded with zeros however far the window reaches.
  const sample = (index) => profile[clamp(index, 0, length - 1)];
  const span = (low, high) =>
    sums[clamp(high, 0, length - 1) + 1] -
    sums[clamp(low, 0, length - 1)] +
    Math.max(0, -low) * profile[0] +
    Math.max(0, high - length + 1) * profile[length - 1];

  for (let index = 0; index < length; index++) {
    out[index] =
      span(index - whole, index + whole) * weight +
      (sample(index - whole - 1) + sample(index + whole + 1)) * edge;
  }
  return out;
}

/**
 * The largest value one axis of the blurred mask reaches.
 *
 * @param {number} core - Width of the opaque core on this axis, in pixels.
 * @param {number} radius - The blur radius, which both filters are run at.
 * @returns {number} The largest value on the axis, 0 through 1.
 */
function axisPeak(core, radius) {
  if (!(core > 0)) return 0;
  if (!(radius > 0)) return 1;

  const box = gaussianBoxRadius(radius);
  const support = Math.ceil(radius) + BLUR_PASSES * (Math.ceil(box) + 1) + 1;
  // An axis whose opaque core is wider than both blurs together reaches one in the middle
  // whatever else is on the axis, which is the ordinary case and is answered without measuring
  // anything.
  if (core / 2 >= support) return 1;

  // Measured at a reduced sampling where the rectangle is larger than the profile is long. The
  // peak is set by the core's width against the blur's, so reducing both leaves it alone.
  const wanted = Math.ceil(core) + 2 * support + 1;
  const reduce = wanted > MAX_SAMPLES ? wanted / MAX_SAMPLES : 1;
  const width = Math.max(1, Math.round(core / reduce));
  const pad = Math.max(1, Math.round(support / reduce));
  const spread = radius / reduce;

  let profile = new Float64Array(width + pad * 2);
  profile.fill(1, pad, pad + width);
  profile = boxPass(profile, spread);
  const step = gaussianBoxRadius(spread);
  for (let pass = 0; pass < BLUR_PASSES; pass++) profile = boxPass(profile, step);

  let most = 0;
  for (let index = 0; index < profile.length; index++) {
    if (profile[index] > most) most = profile[index];
  }
  return most;
}

/**
 * Measure the mask the node would build for one rectangle, keeping the last answer.
 *
 * @param {{width: number, height: number}} size - The rectangle's size in pixels, clamped into
 *   the image as the node clamps it.
 * @param {number} blending - The `crop_blending` widget.
 * @param {object} store - Where the last answer is kept.
 * @returns {{radius: number, inset: number, peak: number}} The blur radius, the border the node
 *   draws, and the largest value the finished mask reaches.
 */
function measureMask(size, blending, store) {
  const key = `${size.width},${size.height},${blending}`;
  if (store.key === key && store.answer) return store.answer;

  const radius = blendRadius(size.width, size.height, blending);
  const feather = featherInset(size.width, size.height, radius);
  // `paste_image` fills a block the size of the rectangle with white, draws a black border
  // `feather_inset` pixels thick just inside that block's own edge, and blurs the result twice
  // at `blend_ratio / 4`, so both blurs run at a quarter of the radius the border was taken
  // from. `BoxBlur` is one pass of a box filter whose radius may be fractional and
  // `GaussianBlur` is three more at the radius Pillow works out for the standard deviation it
  // is given, and both are mirrored here, so the peak below is the peak the mask reaches.
  const spread = radius / 4;
  // What is left between the two borders is the opaque core the blurs have to work with, and
  // wherever that core is narrower than they are wide the mask reaches white nowhere. That
  // begins long before the cap on the border trims anything, so the mask is measured rather
  // than the cap tested: a 512 by 300 paste at a `crop_blending` of 1.0 peaks at 0.21 of white
  // with nothing trimmed at all, and a plain 512 square at the same blending peaks at 0.84.
  const answer = {
    radius,
    inset: feather.inset,
    peak:
      axisPeak(size.width - feather.inset * 2, spread) *
      axisPeak(size.height - feather.inset * 2, spread),
  };

  store.key = key;
  store.answer = answer;
  return answer;
}

/**
 * What the edge of the paste is doing, for the footer's second line.
 *
 * @param {object} node - Node holding the widgets.
 * @param {{width: number, height: number}} frame - The image's size in pixels, zero on either
 *   axis while it is not known.
 * @param {{x: number, y: number, w: number, h: number}|null} rect - The rectangle on screen,
 *   which is the one an unfinished gesture holds while there is one.
 * @param {object} store - Where the last measurement is kept.
 * @returns {string} The line to draw.
 */
function describePaste(node, frame, rect, store) {
  // The rectangle on screen is the one an unfinished gesture holds, so the line follows the
  // drag that is fixing a paste rather than describing the one that was there before it.
  const size = pasteSize(rect ?? readRect(node), frame);
  // A rectangle typed with no area is named here rather than drawn as though it would render,
  // since the node stops with an error on one once it has clamped each edge into the image.
  if (!(size.width >= 1) || !(size.height >= 1)) {
    return "the rectangle has no area, so the node stops";
  }
  // A linked input is read instead of the widget beside it, so the number this line would
  // quote is not the number the run feathers with.
  if (inputLinked(node, CROP_BLENDING)) return "crop_blending is linked";

  const blending = readNumber(node, CROP_BLENDING, DEFAULTS[CROP_BLENDING]);
  const mask = measureMask(size, blending, store);
  if (!(mask.radius > 0)) return "hard edged";

  // That the rectangle is where `crop_image` lands is what the rectangle itself shows, so the
  // feather is what is left for this line to say. Both numbers move with `crop_blending` and
  // with every edge.
  const edge = mask.inset > 0 ? `feather ${mask.inset}px inward` : "feather on the edge";
  // A mask that never reaches white is the whole of a paste that reads faint, and the number is
  // the only account of it the rectangle cannot give.
  const peak = mask.peak >= WHITE ? "" : `, mask peaks at ${mask.peak.toFixed(2)}`;
  return `${edge}${peak}`;
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
 * The published picture, with the size of the image it came from kept for the footer.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {{width: number, height: number}} frame - Record to keep the image's size in.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function trackedBackdrop(node, frame) {
  // The picture is the image the node received, published by `preview.publish` from the node's
  // own `execute` and served by `modules/interface/preview.py`. No fallback frame is stated, so a
  // node that has published nothing draws the stand-in picture and refuses every gesture: there
  // is no frame to measure one in, and a rectangle measured against a guess, the stand-in's own
  // size included, would write numbers that look right on screen and paste somewhere else.
  const source = imageBackdrop(node);
  return {
    async load() {
      const answer = await source.load();
      // `imageBackdrop` answers everything the rectangle needs, and the footer needs one thing
      // more: the size the node clamps the rectangle into. Wrapping the answer on its way past
      // is what gets it, rather than a second request for the same picture.
      const width = Math.max(0, Number(answer?.width) || 0);
      const height = Math.max(0, Number(answer?.height) || 0);
      // An answer stating no size leaves the last one standing, as the editor's own frame does.
      // The store a picture is held in is bounded and evicts the least recently used, so a node
      // in a busy graph answers `waiting` again after it has been answering `ready`, and the
      // image it received on its last run is still the image it received.
      if (width > 0 && height > 0) {
        frame.width = width;
        frame.height = height;
      }
      return answer;
    },
  };
}

/**
 * Chain a repaint onto a widget's callback.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {() => void} onChange - Called after the original callback.
 * @returns {void}
 */
function chainWidgetCallback(node, name, onChange) {
  const widget = findWidget(node, name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a widget change:`, error);
    }
    return result;
  };
}

/**
 * Ask for the picture again as soon as the node finishes, or when a run ends without it.
 *
 * @param {object} node - The node the picture is published for.
 * @param {{refresh: () => void}} editor - Editor from `createRegionEditor`.
 * @returns {() => void} Unhooks the listener.
 */
function watchRuns(node, editor) {
  return onNodeDone(node, () => {
    try {
      editor.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to ask for the image again:`, error);
    }
  });
}

/**
 * Append the editor to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachRegionEditor(node) {
  for (const name of EDGE_NAMES) {
    if (!findWidget(node, name)) return;
  }

  // The size of the image the node received, filled in by the backdrop on its way past and
  // read by the footer alone. Nothing written to a widget passes through it.
  const frame = { width: 0, height: 0 };

  // The last mask measured for the footer, kept so a repaint measures nothing.
  const store = { key: "", answer: null };

  const editor = createRegionEditor({
    node,
    backdrop: trackedBackdrop(node, frame),
    rect: {
      read: () => readRect(node),
      write: (rect, moved) => writeRect(node, rect, moved),
      locks: () => readLocks(node),
    },
    footer: (rect) => describePaste(node, frame, rect, store),
    hover: PASTE_HOVER,
    height: UI_HEIGHT,
    minSize: MIN_SIZE,
  });

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  for (const name of EDGE_NAMES) {
    chainWidgetCallback(node, name, () => editor.handleRectChanged());
  }
  // The blur changes what the footer says about the paste and nothing about the rectangle.
  chainWidgetCallback(node, CROP_BLENDING, () => editor.schedulePaint());

  const stopWatchingRuns = watchRuns(node, editor);

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link
  // changes no widget value, so the callbacks above never hear about it.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      editor.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      editor.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered
  // and its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      stopWatchingRuns();
      editor.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the region editor:`, error);
    }
    return result;
  };

  editor.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Paste Crop by Location", "Region editor"],
      name: "Show the region editor",
      tooltip:
        "Draw the paste rectangle over the image on Image Paste Crop by Location. The widgets " +
        "themselves are always available. This applies to nodes added after the setting " +
        "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second editor.
    if (proto.__was_paste_region_wrapped) return;
    proto.__was_paste_region_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachRegionEditor(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the region editor:`, error);
      }
      return result;
    };
  },
});
