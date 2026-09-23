/**
 * ComfyUI's own mask editor, opened on a mask node and read back into `drawn_mask`.
 *
 * The picture comes from the pack's preview route as an inverted alpha. A save uploads three PNG
 * files into `ComfyUI/input/clipspace`.
 */

import { api } from "../../../scripts/api.js";
import { fetchWithin } from "./request.js";
import { app } from "../../../scripts/app.js";
import { executionId, nodeLocator, placed } from "./preview.js";

const LOG_NAME = "WASNodeSuite.ComfyMaskEditor";

// The command the frontend registers for its own mask editor, and the class its dialog carries.
// That class is the only handle on the dialog's lifetime an extension is given, and a save and
// a cancel both end with the element gone.
const OPEN_COMMAND = "Comfy.MaskEditor.OpenMaskEditor";
const DIALOG_SELECTOR = ".mask-editor-dialog";

// Where the picture comes from. `filename` is present for that editor's loader, and `type`
// names no ComfyUI directory on purpose: the editor sends the pair
// back as the reference its first upload resolves against, and a reference resolving nowhere is
// one file a save does not write.
const PREVIEW_ROUTE = "/was/interface/api/preview";
const SOURCE_FILENAME = "was-node-suite-mask.png";
const SOURCE_TYPE = "preview";

// How long the dialog is waited for before the node is put back, in milliseconds.
const OPEN_TIMEOUT = 10000;

// How far a pixel has to move before it counts as an edit, on a 0 to 255 scale. The selection
// round trips through two PNG encodes and a GPU canvas, and a whole mask shifted by one level
// would otherwise be stored as a drawing of itself.
const EDIT_TOLERANCE = 2;

/**
 * The values of `drawn_combine` a save can be read into.
 */
export const MASK_EDITOR_COMBINES = ["union", "subtract", "off"];

/**
 * What a save does to the drawing, in the words the panel's footer uses.
 *
 * @param {string} mode - What `drawn_combine` holds.
 * @returns {string} One short clause naming the direction a save is read in.
 */
export function maskEditorNote(mode) {
  if (mode === "subtract") return "the editor cuts into the mask";
  if (mode === "intersect") return "no editor on an intersect drawing";
  if (mode === "off") return "the editor adds, the run leaves it out";
  return "the editor adds to the mask";
}

// Remembered once found, since the answer is asked for on every repaint of every panel and a
// command list only grows. A frontend that never registers it is asked again each time, which
// costs one walk of a list a few hundred long and is what lets a late registration be seen.
let commandSeen = false;

/**
 * Whether the frontend offers its mask editor at all.
 *
 * @returns {boolean} True while the command that opens it is registered.
 */
function commandOffered() {
  if (commandSeen) return true;
  try {
    const commands = app?.extensionManager?.command?.commands;
    commandSeen = Array.isArray(commands) && commands.some((entry) => entry?.id === OPEN_COMMAND);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read the frontend's commands:`, error);
    return false;
  }
  return commandSeen;
}

/**
 * Say that a save went nowhere, where the person who pressed save will see it.
 *
 * @param {string} detail - The sentence to show.
 * @returns {void}
 */
function report(detail) {
  console.warn(`[${LOG_NAME}] ${detail}`);
  try {
    app?.extensionManager?.toast?.add?.({
      severity: "warn",
      summary: "Mask editor",
      detail,
      life: 8000,
    });
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to show a message:`, error);
  }
}

/**
 * The address the editor loads a node's published mask from.
 *
 * @param {object} node - The node the mask was published by.
 * @returns {string} The preview route with the parameters that loader requires, or the empty
 *   string for a node carrying no execution id.
 */
function previewUrl(node) {
  const id = executionId(node);
  if (!placed(id)) return "";
  return api.apiURL(
    `${PREVIEW_ROUTE}?node_id=${encodeURIComponent(id)}&side=output`
      + `&filename=${encodeURIComponent(SOURCE_FILENAME)}`
      + `&type=${encodeURIComponent(SOURCE_TYPE)}`,
  );
}

/** Route that deletes the uploads a save made, which this interface keeps none of. */
const DISCARD_ROUTE = "/was/interface/api/preview/discard";

/** The layers that editor writes, all under one stamp, all named from the reference it hands back. */
const UPLOAD_LAYERS = ["mask", "paint", "painted", "painted-masked"];

/**
 * Delete the pictures a save uploaded.
 *
 * @param {Array<object>} value - What the editor assigned to `node.images`.
 * @returns {void} Nothing is awaited: a save must not wait on a tidy-up, and a file left behind
 *   costs a menu entry rather than the edit.
 */
function discardUploads(value) {
  try {
    const first = (Array.isArray(value) ? value : [value]).find((entry) => entry?.filename);
    const stamp = String(first?.filename || "").match(/-(\d{6,})\.png$/)?.[1];
    if (!stamp) return;
    const names = UPLOAD_LAYERS.map((layer) => `clipspace-${layer}-${stamp}.png`);
    fetchWithin(DISCARD_ROUTE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    }).catch(() => {});
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to discard the editor's uploads:`, error);
  }
}

/**
 * Hold a node's picture fields for one visit to the editor, capturing what a save writes.
 *
 * @param {object} node - The node the editor is opened on.
 * @param {object} seed - What `node.imgs` answers while the editor is open.
 * @param {(picture: object) => void} onSaved - Called with the picture a save assigns.
 * @param {() => void} onLanded - Called once the save has assigned its uploaded references,
 *   which is the frontend's own mark that every upload went through.
 * @returns {{release: () => void}} Puts every field back the way it was found.
 */
function holdNode(node, seed, onSaved, onLanded) {
  const hadImgs = Object.prototype.hasOwnProperty.call(node, "imgs");
  const heldImgs = node.imgs;
  const hadImages = Object.prototype.hasOwnProperty.call(node, "images");
  const heldImages = node.images;
  const hadDraw = Object.prototype.hasOwnProperty.call(node, "onDrawBackground");
  const heldDraw = node.onDrawBackground;
  const hadHide = Object.prototype.hasOwnProperty.call(node, "hideOutputImages");
  const heldHide = node.hideOutputImages;

  const locator = nodeLocator(node);
  const outputs = app?.nodeOutputs ?? null;
  const hadOutput = !!locator && !!outputs
    && Object.prototype.hasOwnProperty.call(outputs, locator);
  const heldOutput = hadOutput ? outputs[locator] : undefined;

  let imgs = [seed];
  let images;

  // The node's own draw handler adds an image preview widget to any node holding `imgs` and
  // never takes it away, so it is shut off for the visit rather than left to grow one over the
  // panel. The rest of that handler is for subgraph nodes, which these four are not.
  node.onDrawBackground = () => {};
  // The Vue node renderer reads this before it reads the output store, so it is what stops an
  // uploaded clipspace picture appearing on the node between a save and the cleanup below.
  node.hideOutputImages = true;
  // The editor's loader prefers an entry here over `node.imgs`, so an entry left by an earlier
  // visit would send it to a file in `input/clipspace` instead of the mask this node produced.
  if (hadOutput) delete outputs[locator];

  Object.defineProperty(node, "imgs", {
    configurable: true,
    enumerable: true,
    get: () => imgs,
    set: (value) => {
      imgs = value;
      // A save assigns the painted and masked canvas here first, then clears the field once its
      // uploads have landed. Only the assignment carrying a picture is the save.
      const picture = Array.isArray(value) ? value[0] : null;
      if (picture) onSaved(picture);
    },
  });

  Object.defineProperty(node, "images", {
    configurable: true,
    enumerable: true,
    get: () => images,
    set: (value) => {
      images = value;
      if (value) {
        discardUploads(value);
        onLanded();
      }
    },
  });

  return {
    release() {
      delete node.imgs;
      delete node.images;
      if (hadImgs) node.imgs = heldImgs;
      if (hadImages) node.images = heldImages;
      if (outputs && locator) {
        if (hadOutput) outputs[locator] = heldOutput;
        else delete outputs[locator];
      }
      if (hadDraw) node.onDrawBackground = heldDraw;
      else delete node.onDrawBackground;
      if (hadHide) node.hideOutputImages = heldHide;
      else delete node.hideOutputImages;
    },
  };
}

/**
 * Watch the frontend's mask editor dialog through one visit.
 *
 * @param {() => void} onClosed - Called once, when the dialog has gone or never arrived.
 * @returns {{stop: () => void}} Stops watching without ever calling back.
 */
function watchDialog(onClosed) {
  let seen = false;
  let done = false;
  let timer = 0;
  let observer = null;

  const settle = () => {
    if (done) return false;
    done = true;
    observer?.disconnect();
    if (timer) clearTimeout(timer);
    timer = 0;
    return true;
  };

  const look = () => {
    if (done) return;
    if (document.querySelector(DIALOG_SELECTOR)) {
      seen = true;
      // The dialog is here, so the wait for it to arrive is over and only its removal ends the
      // visit from now on.
      if (timer) clearTimeout(timer);
      timer = 0;
      return;
    }
    if (!seen) return;
    if (settle()) onClosed();
  };

  observer = new MutationObserver(look);
  observer.observe(document.body, { childList: true, subtree: true });
  timer = setTimeout(() => {
    if (seen) return;
    if (settle()) onClosed();
  }, OPEN_TIMEOUT);
  look();

  return {
    stop() {
      settle();
    },
  };
}

/**
 * Read the selection out of the picture a save handed back.
 *
 * @param {object} picture - The painted and masked canvas, whose alpha is 255 minus the mask.
 * @param {number} width - Columns expected.
 * @param {number} height - Rows expected.
 * @returns {Uint8ClampedArray|null} The selection, one level a pixel, or null when the picture
 *   cannot be read.
 */
function selectionLevels(picture, width, height) {
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(picture, 0, 0, width, height);

  const pixels = ctx.getImageData(0, 0, width, height).data;
  const levels = new Uint8ClampedArray(width * height);
  for (let at = 0; at < levels.length; at++) levels[at] = 255 - pixels[at * 4 + 3];
  return levels;
}

/**
 * What a save contributes to the drawing, at the picture's own size.
 *
 * @param {{width: number, height: number, levels: Uint8ClampedArray}} plane - The computed mask
 *   the editor opened on.
 * @param {Uint8ClampedArray} selection - The selection the save handed back.
 * @param {string} mode - What `drawn_combine` holds.
 * @returns {Float32Array|null} Coverage from 0 to 1, or null when the save moved nothing the
 *   run can carry.
 */
function contribution(plane, selection, mode) {
  const values = new Float32Array(plane.levels.length);
  let any = false;
  for (let at = 0; at < values.length; at++) {
    const computed = plane.levels[at];
    const chosen = selection[at];
    // `subtract` takes the drawing away from the mask, so the drop below the computed level is
    // the part of the selection the run can carry. Every other offered combine adds, and there
    // the level chosen is stored rather than the difference, so a soft mask painted solid
    // arrives solid rather than at the gap between the two.
    const moved = mode === "subtract"
      ? (computed > chosen + EDIT_TOLERANCE ? (computed - chosen) / 255 : 0)
      : (chosen > computed + EDIT_TOLERANCE ? chosen / 255 : 0);
    if (moved > 0) any = true;
    values[at] = moved;
  }
  return any ? values : null;
}

/**
 * Build the canvas shape a drawing is held in.
 *
 * @param {Float32Array} values - Coverage from 0 to 1, row major.
 * @param {number} width - Columns.
 * @param {number} height - Rows.
 * @returns {HTMLCanvasElement|null} White everywhere with the coverage in the alpha, which is
 *   what the brush holds its strokes as, or null when it cannot be drawn into.
 */
function coverageCanvas(values, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const image = ctx.createImageData(width, height);
  const pixels = image.data;
  for (let at = 0; at < values.length; at++) {
    const pixel = at * 4;
    pixels[pixel] = 255;
    pixels[pixel + 1] = 255;
    pixels[pixel + 2] = 255;
    pixels[pixel + 3] = Math.round(values[at] * 255);
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

/**
 * Draw coverage onto a grid of another size.
 *
 * @param {Float32Array} values - Coverage from 0 to 1, row major.
 * @param {{width: number, height: number}} from - The size the values are at.
 * @param {{width: number, height: number}} to - The size wanted.
 * @returns {Float32Array|null} Coverage at the size wanted, or null when it cannot be drawn.
 */
function resample(values, from, to) {
  if (from.width === to.width && from.height === to.height) return values;
  const source = coverageCanvas(values, from.width, from.height);
  if (!source) return null;

  const scratch = document.createElement("canvas");
  scratch.width = to.width;
  scratch.height = to.height;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, to.width, to.height);

  const pixels = ctx.getImageData(0, 0, to.width, to.height).data;
  const answer = new Float32Array(to.width * to.height);
  for (let at = 0; at < answer.length; at++) answer[at] = pixels[at * 4 + 3] / 255;
  return answer;
}

/**
 * Build the bridge to the frontend's mask editor for one node.
 *
 * @param {object} options - What the editor opens on and where its answer goes.
 * @param {object} options.node - The node the editor is opened on.
 * @param {() => {width: number, height: number, levels: Uint8ClampedArray}|null}
 *   options.computed - The mask the node's last run produced, at the size the preview route
 *   served it.
 * @param {() => {width: number, height: number}} options.storeSize - The size a drawing is held
 *   at, from `maskStoreSize`.
 * @param {(width: number, height: number) => Float32Array|null} options.sample - The drawing
 *   already held, reduced to a given size.
 * @param {(canvas: HTMLCanvasElement) => void} options.adopt - Stores a drawing, as one edit.
 * @param {() => string} options.combine - What the node's `drawn_combine` widget holds.
 * @returns {{available: () => boolean, offered: () => boolean, open: () => boolean,
 *   busy: () => boolean, dispose: () => void}} Whether the frontend has that editor and the
 *   node has a mask for it, whether the combine in force can carry a save as well, the way in,
 *   whether a visit is in progress, and teardown.
 */
export function createComfyMaskEditor(options = {}) {
  const settings = {
    node: options.node ?? null,
    computed: typeof options.computed === "function" ? options.computed : () => null,
    storeSize: typeof options.storeSize === "function"
      ? options.storeSize
      : () => ({ width: 0, height: 0 }),
    sample: typeof options.sample === "function" ? options.sample : () => null,
    adopt: typeof options.adopt === "function" ? options.adopt : () => {},
    combine: typeof options.combine === "function" ? options.combine : () => "union",
  };

  const state = { visit: null, disposed: false };

  /**
   * Whether this frontend has that editor and this node has a mask to hand it.
   *
   * @returns {boolean} True while the command is registered and the node has published.
   */
  function available() {
    if (state.disposed || !settings.node) return false;
    if (!settings.computed()) return false;
    return commandOffered();
  }

  /**
   * Whether the editor can be opened as things stand.
   *
   * @returns {boolean} True while it is available and `drawn_combine` is one a save can be
   *   read into.
   */
  function offered() {
    return available() && MASK_EDITOR_COMBINES.includes(settings.combine());
  }

  /**
   * Join what a save carried into the drawing already held, and store the pair.
   *
   * @param {object} picture - The picture the save assigned.
   * @param {string} mode - The `drawn_combine` in force when the editor was opened, which is
   *   the direction the chip named at the time.
   * @returns {void}
   */
  function absorb(picture, mode) {
    const plane = settings.computed();
    if (!plane) return;

    const width = Math.max(0, picture?.naturalWidth || picture?.width || 0);
    const height = Math.max(0, picture?.naturalHeight || picture?.height || 0);
    if (width !== plane.width || height !== plane.height) {
      report(
        `The mask editor saved a ${width}x${height} picture and this node's mask is `
          + `${plane.width}x${plane.height}. A rotation inside the editor cannot be read back `
          + "onto the mask, so the drawing was left as it was.",
      );
      return;
    }

    const selection = selectionLevels(picture, width, height);
    if (!selection) return;

    const values = contribution(plane, selection, mode);
    if (!values) {
      report(
        `Nothing in that save moved the mask in the direction drawn_combine reads: with ${mode}, `
          + `${maskEditorNote(mode)}. The drawing was left as it was.`,
      );
      return;
    }

    const size = settings.storeSize();
    if (!(size.width > 0) || !(size.height > 0)) return;
    const scaled = resample(values, plane, size);
    if (!scaled) return;

    // The drawing already held is joined in rather than replaced, and the join is the larger of
    // the two at each pixel, which is what `combine` in `modules/mask/drawn.py` does for a
    // union. Adding them instead would make the overlap of two soft edges brighter than either.
    const held = settings.sample(size.width, size.height);
    if (held && held.length === scaled.length) {
      for (let at = 0; at < scaled.length; at++) {
        if (held[at] > scaled[at]) scaled[at] = held[at];
      }
    }

    const canvas = coverageCanvas(scaled, size.width, size.height);
    if (canvas) settings.adopt(canvas);
  }

  /**
   * End a visit: put the node back, then store whatever the save handed over.
   *
   * @param {object} visit - The visit being ended.
   * @returns {void}
   */
  function close(visit) {
    if (state.visit !== visit) return;
    state.visit = null;
    visit.watch?.stop();
    const captured = visit.captured;
    try {
      visit.hold?.release();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to put the node back:`, error);
    }
    // Both marks are wanted. The picture is assigned before the uploads and the references
    // after them, so a picture on its own is a save that raised part way through, and the
    // dialog it left open is the one the person is still looking at.
    if (!captured || !visit.saved || state.disposed) return;
    try {
      absorb(captured, visit.mode);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to store what the mask editor saved:`, error);
    }
  }

  return {
    available,
    offered,

    /**
     * Open the editor on this node's published mask.
     *
     * @returns {boolean} True when the editor was asked for. False leaves the node untouched.
     */
    open() {
      if (state.disposed || state.visit || !offered()) return false;
      const node = settings.node;
      const url = previewUrl(node);
      if (!url) return false;

      const seed = new Image();
      seed.src = url;

      // Read once, here, so a save is stored the way the chip said it would be even if the
      // combo is changed on the canvas behind the dialog.
      const visit = {
        captured: null,
        saved: false,
        mode: settings.combine(),
        hold: null,
        watch: null,
      };
      state.visit = visit;
      visit.hold = holdNode(
        node,
        seed,
        (picture) => {
          if (!visit.captured) visit.captured = picture;
        },
        () => {
          visit.saved = true;
        },
      );
      visit.watch = watchDialog(() => close(visit));

      try {
        app.canvas?.selectNode?.(node);
        const running = app.extensionManager.command.execute(OPEN_COMMAND);
        running?.catch?.((error) => {
          console.error(`[${LOG_NAME}] The mask editor did not open:`, error);
          close(visit);
        });
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to open the mask editor:`, error);
        close(visit);
        return false;
      }
      return true;
    },

    /**
     * Whether a visit is in progress.
     *
     * @returns {boolean} True between the open and the dialog going away.
     */
    busy() {
      return state.visit !== null;
    },

    /**
     * Release the bridge, putting the node back when a visit is still open.
     *
     * @returns {void}
     */
    dispose() {
      state.disposed = true;
      const visit = state.visit;
      if (!visit) return;
      state.visit = null;
      visit.watch?.stop();
      try {
        visit.hold?.release();
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to put the node back:`, error);
      }
    },
  };
}
