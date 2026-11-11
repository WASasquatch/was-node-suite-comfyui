/**
 * The mask a node computed, drawn under its widgets with the chip that corrects it.
 *
 * The `editor` chip opens ComfyUI's mask editor on the published mask, and a save is joined
 * into `drawn_mask`.
 */

import { app } from "../../scripts/app.js";
import { createComfyMaskEditor, maskEditorNote } from "./interface/comfy_mask_editor.js";
import { combine, createMaskPaint } from "./interface/mask_paint.js";
import { maskStoreSize, maskValueBytes } from "./interface/mask_value.js";
import { PREVIEW_SIDE, PREVIEW_STATE, fetchOutputPreview } from "./interface/preview.js";
import { createRegionEditor } from "./interface/region.js";
import { readableBytes } from "./interface/report_panel.js";
import { onNodeFinished } from "./interface/run_events.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.MaskBrush";
const SETTING_ID = "WAS.Mask.ShowBrush";

// The mask nodes that carry the pair of drawing widgets. A node absent from this list and a
// node whose python is older than this file both end up the same way: no panel, plain widgets.
const NODES = [
  "Mask Threshold Region",
  "Mask Fill Holes",
  "Mask Dominant Region",
  "Mask Minority Region",
];

const DRAWN_MASK_WIDGET = "drawn_mask";
const DRAWN_COMBINE_WIDGET = "drawn_combine";

// What `drawn_combine` holds on a node whose widget cannot be read, from `DEFAULT_COMBINE` in
// `modules/mask/drawn.py`.
const DEFAULT_COMBINE = "union";

const UI_WIDGET_NAME = "was_mask_brush_ui";
const UI_WIDGET_TYPE = "was_mask_brush_region";

// Height of the panel in node units. The widget's margin, the region editor's padding and its
// two footer lines spend 56 of them, which leaves 72 element pixels of picture. The chip row is
// 18 of those, so the one chip reads as a badge on the top left corner of the mask rather than a
// lid over it, and nothing is aimed at the rest: it is there to be judged, not drawn on. The
// region editor's maximum is unbounded, so the node's spare room reaches this panel as soon as
// the node is dragged taller.
const PANEL_HEIGHT = 128;

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
 * Read a widget's value as text.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {string} The widget's value, empty when it is absent or holds no string.
 */
function readText(node, name) {
  const value = findWidget(node, name)?.value;
  return typeof value === "string" ? value : "";
}

/**
 * Write a widget's value as text, once.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {string} value - Value to store.
 * @returns {void}
 */
function writeText(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget) return;
  if (widget.value === value) return;
  // A single line string widget is a plain widget whose value setter runs no callback, so the
  // repaint that follows is the caller's rather than something this write sets off.
  widget.value = value;
}

/**
 * What the node does with the drawing.
 *
 * @param {object} node - Node holding the widget.
 * @returns {string} The combine, falling back to the schema default while the widget is absent.
 */
function readCombine(node) {
  return readText(node, DRAWN_COMBINE_WIDGET) || DEFAULT_COMBINE;
}

/**
 * Read whether the panel is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function panelEnabled() {
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
 * Read one channel of a picture into a plane of levels.
 *
 * @param {HTMLImageElement} picture - The decoded thumbnail, mode L on the wire so every colour
 *   channel carries the same level.
 * @returns {{width: number, height: number, levels: Uint8ClampedArray}|null} The mask at the
 *   picture's own size, one byte a pixel, or null when it cannot be read.
 */
function readLevels(picture) {
  const width = Math.max(0, picture?.naturalWidth || picture?.width || 0);
  const height = Math.max(0, picture?.naturalHeight || picture?.height || 0);
  if (!(width > 0) || !(height > 0)) return null;

  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(picture, 0, 0, width, height);

  const pixels = ctx.getImageData(0, 0, width, height).data;
  // One byte a pixel rather than four, since a mask is one channel and the plane is held for the
  // life of the run's picture while every stroke composes against it.
  const levels = new Uint8ClampedArray(width * height);
  for (let at = 0; at < levels.length; at++) levels[at] = pixels[at * 4];
  return { width, height, levels };
}

/**
 * Compose the computed mask and the drawing into the picture the run will produce.
 *
 * @param {object} plane - The computed mask, from `readLevels`.
 * @param {Float32Array} drawn - The drawing reduced to the plane's size, 0 to 1.
 * @param {string} mode - The combine in force, one of the modes `modules/mask/drawn.py` names.
 * @param {HTMLCanvasElement} target - Canvas to draw into, resized to the plane.
 * @returns {HTMLCanvasElement|null} The canvas, or null when it cannot be drawn into.
 */
function composeMask(plane, drawn, mode, target) {
  target.width = plane.width;
  target.height = plane.height;
  const ctx = target.getContext("2d");
  if (!ctx) return null;

  const image = ctx.createImageData(plane.width, plane.height);
  const pixels = image.data;
  for (let at = 0; at < plane.levels.length; at++) {
    const level = Math.round(combine(plane.levels[at] / 255, drawn[at], mode) * 255);
    const pixel = at * 4;
    pixels[pixel] = level;
    pixels[pixel + 1] = level;
    pixels[pixel + 2] = level;
    pixels[pixel + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return target;
}

/**
 * The backdrop: the mask the node's last run produced, with the drawing joined into it.
 *
 * @param {object} node - Node the panel is drawn on.
 * @param {object} store - Where the fetched answer, its levels and the composed canvas are kept.
 * @param {() => object|null} paint - Answers the store the drawing is held in, once it exists.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function maskBackdrop(node, store, paint) {
  return {
    async load() {
      // Refetched when a run has finished and when the last answer carried no picture. A
      // repaint driven by a save reaches neither, so an edit composes against the plane
      // already held instead of asking the server once per commit.
      if (store.stale || !store.answer || store.answer.state !== PREVIEW_STATE.READY) {
        store.answer = await fetchOutputPreview(node);
        store.stale = false;
        store.plane = null;
        store.composedKey = "";
      }

      const answer = store.answer;
      if (answer?.state !== PREVIEW_STATE.READY) {
        // No size at all rather than a remembered one: the mask's size is a fact about the run
        // and an edit made against a guess would land somewhere else on the next.
        return { ...answer, width: 0, height: 0, scale: 1, frameSource: PREVIEW_SIDE.OUTPUT };
      }

      const frame = {
        ...answer,
        width: answer.sourceWidth,
        height: answer.sourceHeight,
        frameSource: PREVIEW_SIDE.OUTPUT,
      };

      if (!store.plane) store.plane = readLevels(answer.image);
      const plane = store.plane;
      if (!plane) return frame;

      const drawing = paint();
      const mode = readCombine(node);
      // `off` is what the run reads, so the picture leaves the drawing out as well and the
      // layer over it is what says the drawing is still there.
      const drawn = mode === "off" ? null : drawing?.sample(plane.width, plane.height);
      if (!drawn) return frame;

      const key = `${mode}:${drawing.version()}`;
      if (store.composedKey !== key) {
        if (!store.composed) store.composed = document.createElement("canvas");
        if (!composeMask(plane, drawn, mode, store.composed)) return frame;
        store.composedKey = key;
      }
      return { ...frame, image: store.composed };
    },
  };
}

/**
 * What the drawing is doing, on the footer's second line.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} store - Where the fetched answer is kept.
 * @param {object|null} drawing - The store the drawing is held in, asked what it holds.
 * @param {object|null} bridge - The way to the frontend's mask editor, asked whether it is
 *   there, so the line names a save's direction only where a save can be made.
 * @returns {string} The line to draw.
 */
function footerMeaning(node, store, drawing, bridge) {
  const answer = store.answer;
  // Read the same way `maskSize` reads it, so the line agrees with whether the chip is usable.
  // No answer yet is no known mask size, which is the state the editor cannot be opened in, and
  // measuring a stored drawing against a size of nothing reports a resize on every run.
  if (answer?.state !== PREVIEW_STATE.READY) {
    return "run the node once to correct the mask it makes";
  }

  const mode = readCombine(node);
  // Which way a save from the editor is read is the one thing about it that can be got wrong
  // without noticing, so it is on the panel rather than only in the hover text.
  const note = bridge?.available?.() ? `, ${maskEditorNote(mode)}` : "";

  const drawn = drawing?.header();
  if (!drawn) return `nothing drawn${note}`;

  const bytes = readableBytes(maskValueBytes(readText(node, DRAWN_MASK_WIDGET)));
  const mask = maskStoreSize(answer.sourceWidth, answer.sourceHeight);
  // The size the drawing was made at is only worth the room when it is not the size the mask is
  // now, since that is the case where the run resizes it and every pixel of it has moved.
  if (drawn.width === mask.width && drawn.height === mask.height) {
    return `${mode} drawing, ${bytes}${note}`;
  }
  return `${mode} drawing made at ${drawn.width}x${drawn.height}, resized to the mask, `
    + `${bytes}${note}`;
}

/**
 * Where a correction is made and what it costs, for the footer's hover text.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} store - Where the fetched answer is kept.
 * @param {object|null} bridge - The way to the frontend's mask editor, asked whether it can be
 *   opened as things stand, since a panel with no chip on it has to say why.
 * @returns {string} The sentence.
 */
function footerHover(node, store, bridge) {
  const answer = store.answer;
  if (answer?.state !== PREVIEW_STATE.READY) {
    return (
      "The mask this node makes is the picture the editor opens on, and its size is only known "
      + "once the node has run. Queue the graph and the editor chip appears."
    );
  }
  const mask = `The ${answer.sourceWidth}x${answer.sourceHeight} mask this node produced is `
    + "drawn here. What the editor saves is stored in the drawn_mask widget and joined to the "
    + "mask on the next run, in whichever way drawn_combine names, so it saves with the "
    + "workflow and runs the same with no browser open. Clearing that widget removes it.";
  if (bridge?.offered?.()) {
    return `${mask} The editor chip opens ComfyUI's own mask editor on that same mask, with it `
      + "already selected. A save is joined into what is stored rather than replacing it, so "
      + "the drawing only ever grows and one undo takes a save back.";
  }
  if (bridge?.available?.()) {
    return `${mask} No editor while drawn_combine is ${readCombine(node)}: there the drawing is `
      + "the whole region kept, so a save would replace that region rather than extend it. Set "
      + "drawn_combine to union, subtract or off for the editor chip.";
  }
  // Either the frontend registers no such editor or the picture could not be read, and the one
  // thing worth saying is the same in both: the widget is what is left.
  return `${mask} No mask editor is offered here, and that widget is the only way to change `
    + "the drawing.";
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
 * Append the panel to a node and wire it to the widgets it writes.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPanel(node) {
  // A frontend newer than the python beside it reaches a node with no drawing widget, and that
  // node has nothing for a save to write into.
  if (!findWidget(node, DRAWN_MASK_WIDGET)) return;

  const store = {
    answer: null,
    plane: null,
    composed: null,
    composedKey: "",
    stale: true,
    refreshHandle: 0,
    disposed: false,
  };

  /**
   * The mask's own size, which is the frame and the canvas alike.
   *
   * @returns {{width: number, height: number}} The size in pixels, zero until the node has run.
   */
  function maskSize() {
    const answer = store.answer;
    if (answer?.state !== PREVIEW_STATE.READY) return { width: 0, height: 0 };
    return { width: answer.sourceWidth, height: answer.sourceHeight };
  }

  /**
   * The computed mask as a plane of levels, at the size the preview route served it.
   *
   * @returns {{width: number, height: number, levels: Uint8ClampedArray}|null} The plane the
   *   backdrop is built from, or null before the node has published one.
   */
  function computedPlane() {
    const answer = store.answer;
    if (answer?.state !== PREVIEW_STATE.READY) return null;
    if (!store.plane) store.plane = readLevels(answer.image);
    return store.plane;
  }

  /**
   * The chip row, which is the one chip that opens the frontend's mask editor.
   *
   * @returns {Array<object>} One chip while that editor can be opened and the combine in force
   *   can carry what it saves, and none otherwise.
   */
  function editorChips() {
    if (!bridge?.offered?.()) return [];
    const mask = maskSize();
    return [
      {
        key: "editor",
        label: "editor",
        title:
          `Open ComfyUI's mask editor on the ${mask.width}x${mask.height} mask this node `
          + "produced, with that mask already selected. A save is joined into what is stored "
          + `rather than replacing it, so ${maskEditorNote(readCombine(node))}, the drawing `
          + "only ever grows and one undo takes a save back.",
        press: () => bridge?.open?.(),
      },
    ];
  }

  // Named before the drawing so the chip list can ask it, and built after, since the way a save
  // is stored runs through that store's own commit.
  let bridge = null;

  // No mode, so no brush: a save from ComfyUI's editor is the only thing that writes here. That
  // leaves no `frame` and no `canvas` either, since those two convert the position and the
  // radius of a stroke. What remains is the store, the layer that tints it over the picture, and
  // the `sample` and `adopt` pair `createComfyMaskEditor` reads and writes it through.
  const paint = createMaskPaint({
    value: {
      read: () => readText(node, DRAWN_MASK_WIDGET),
      write: (value) => writeText(node, DRAWN_MASK_WIDGET, value),
    },
    combine: () => readCombine(node),
    modes: [],
    actions: editorChips,
  });

  bridge = createComfyMaskEditor({
    node,
    computed: computedPlane,
    storeSize: () => {
      const mask = maskSize();
      return maskStoreSize(mask.width, mask.height);
    },
    sample: (width, height) => paint.sample(width, height),
    adopt: (canvas) => paint.adopt(canvas),
    combine: () => readCombine(node),
  });

  const editor = createRegionEditor({
    node,
    backdrop: maskBackdrop(node, store, () => paint),
    // No `rect`: the mask is the node's own work and there is no box on it to drag.
    footer: () => footerMeaning(node, store, paint, bridge),
    hover: () => footerHover(node, store, bridge),
    height: PANEL_HEIGHT,
    layers: [paint.layer],
    // The chip row rather than the brush. The pointer opens the editor and does nothing else
    // over this picture.
    tool: paint.chips,
  });

  /**
   * Compose the picture again, once, on the next frame.
   *
   * @returns {void}
   */
  function scheduleBackdrop() {
    if (store.disposed || store.refreshHandle) return;
    store.refreshHandle = requestAnimationFrame(() => {
      store.refreshHandle = 0;
      if (store.disposed) return;
      try {
        editor.refresh();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the mask again:`, error);
      }
    });
  }

  paint.bind(editor, () => scheduleBackdrop());
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // The node publishes its mask as it finishes, so the picture is asked for then rather than
  // polled. `executed` is never used for this: it is inverted for nodes like these, sent for a
  // cached node and withheld from one that produced no `ui`.
  const stopWatching = onNodeFinished(node, () => {
    store.stale = true;
    scheduleBackdrop();
  });

  // The drawing is joined into the picture, so a hand edit of either widget composes it again.
  // A save does not reach these: a single line string widget runs no callback when its value is
  // assigned, so the store asks for its own repaint.
  for (const name of [DRAWN_MASK_WIDGET, DRAWN_COMBINE_WIDGET]) {
    chainWidgetCallback(node, name, () => {
      paint.invalidate();
      editor.schedulePaint();
      scheduleBackdrop();
    });
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      // A workflow load and an undo both arrive here, and both can replace the drawing with a
      // different one, so the decoded copy is dropped rather than drawn over the new value.
      paint.invalidate();
      editor.schedulePaint();
      scheduleBackdrop();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      store.disposed = true;
      if (store.refreshHandle) cancelAnimationFrame(store.refreshHandle);
      store.refreshHandle = 0;
      store.plane = null;
      store.composed = null;
      stopWatching();
      // Before the store, since a visit still open puts the node's picture fields back here.
      bridge?.dispose?.();
      editor.dispose();
      paint.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the mask panel:`, error);
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
      category: ["WAS Node Suite", "Masking", "Correct the mask a node makes"],
      name: "Show the mask panel",
      tooltip:
        "Draw the mask made by Mask Threshold Region, Mask Fill Holes, Mask Dominant Region "
        + "and Mask Minority Region under their widgets, with a chip that opens ComfyUI's own "
        + "mask editor on it. What that editor saves is written into the drawn_mask widget and "
        + "joins the mask in whichever way drawn_combine names, so it saves with the workflow "
        + "and runs the same with no browser open. The node has to have run once before there "
        + "is a mask to correct. This applies to nodes added after the setting changes, so a "
        + "reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Node definitions are registered again on a definitions refresh, which would otherwise wrap
    // the prototype a second time and append a second panel.
    if (proto.__was_mask_brush_wrapped) return;
    proto.__was_mask_brush_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (panelEnabled()) attachPanel(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the mask panel:`, error);
      }
      return result;
    };
  },
});
