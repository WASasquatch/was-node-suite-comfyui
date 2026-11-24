/**
 * A record of every gesture made on a region editor.
 *
 * Off unless `WAS.Region.Trace` is on. Each press to release is kept in memory and read with
 * `WASRegionTrace.snaps()`.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.RegionTrace";
const SETTING_ID = "WAS.Region.Trace";

/** Node ids carrying a region editor. */
const NODES = ["Mask Rect Area", "Mask Rect Area (Advanced)", "Image Crop Location", "Image Paste Crop by Location"];

/** Widgets whose movement is worth recording, by name. */
const TRACKED = ["x", "y", "width", "height", "image_width", "image_height", "blur_radius", "top", "left", "right", "bottom"];

/** Gestures kept before the oldest is dropped, so a long session cannot grow without bound. */
const MAX_ENTRIES = 200;

/** Pointer travel, in element pixels, at or under which a gesture reads as a click. */
const CLICK_TRAVEL = 6;

/** Movement in a widget, in its own units, at or over which a click-sized gesture is a jump. */
const JUMP = 6;

const entries = [];

/**
 * Whether tracing is on.
 *
 * @returns {boolean} True only when the setting is explicitly on, since this costs a listener
 *   on every region editor and is a diagnostic rather than a feature.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : false;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return false;
  }
}

/**
 * The tracked widget values of one node.
 *
 * @param {object} node - The node the editor is drawn on.
 * @returns {object} Widget name to value, for the widgets this records.
 */
function values(node) {
  const out = {};
  for (const widget of node.widgets ?? []) {
    if (TRACKED.includes(widget.name)) out[widget.name] = widget.value;
  }
  return out;
}

/**
 * The editor canvas's backing store against the size it is drawn at.
 *
 * @param {HTMLElement} element - The editor's host element.
 * @returns {object|null} The backing store and the drawn size, or null when there is no canvas.
 */
function surfaceOf(element) {
  const canvas = element.querySelector("canvas");
  if (!canvas) return null;
  const box = canvas.getBoundingClientRect();
  return {
    backing: [canvas.width, canvas.height],
    drawn: [Math.round(box.width), Math.round(box.height)],
  };
}

/**
 * Where a pointer landed, in element pixels and in the frame's own percentage.
 *
 * @param {HTMLElement} element - The editor's host element.
 * @param {PointerEvent} event - The event to place.
 * @returns {object} The position, the element's box, and the position as a percentage.
 */
function place(element, event) {
  const box = element.getBoundingClientRect();
  const side = Math.min(box.width, box.height);
  const x = event.clientX - box.x;
  const y = event.clientY - box.y;
  return {
    x: Number(x.toFixed(1)),
    y: Number(y.toFixed(1)),
    box: { width: Math.round(box.width), height: Math.round(box.height) },
    percent: [
      Number((((x - (box.width - side) / 2) / side) * 100).toFixed(1)),
      Number((((y - (box.height - side) / 2) / side) * 100).toFixed(1)),
    ],
  };
}

/**
 * Watch one node's editor, once.
 *
 * @param {object} node - The node to watch.
 * @returns {boolean} True when a listener was added, false when there is no editor yet or one
 *   is already watched.
 */
function watch(node) {
  const element = (node.widgets ?? []).find((widget) => widget.element)?.element;
  if (!element || element.dataset.wasRegionTrace === "on") return false;
  element.dataset.wasRegionTrace = "on";

  let open = null;

  // Captured rather than bubbled, so the record is taken even if the editor stops the event.
  element.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    open = {
      node: node.id,
      type: node.type,
      zoom: Number((app.canvas?.ds?.scale ?? 1).toFixed(3)),
      surface: surfaceOf(element),
      pointerType: event.pointerType,
      down: place(element, event),
      before: values(node),
      moves: 0,
    };
  }, true);

  element.addEventListener("pointermove", (event) => {
    if (!open || !(event.buttons & 1)) return;
    open.moves += 1;
    open.up = place(element, event);
  }, true);

  element.addEventListener("pointerup", () => {
    if (!open) return;
    const record = open;
    open = null;
    // The widgets are written as the gesture ends, so the after reading is taken on the next
    // turn of the event loop rather than in this handler.
    setTimeout(() => {
      record.after = values(node);
      record.travel = record.up
        ? Number(Math.hypot(record.up.x - record.down.x, record.up.y - record.down.y).toFixed(1))
        : 0;
      record.moved = {};
      for (const name of Object.keys(record.after)) {
        const change = (record.after[name] ?? 0) - (record.before[name] ?? 0);
        if (change !== 0) record.moved[name] = change;
      }
      const largest = Math.max(0, ...Object.values(record.moved).map(Math.abs));
      record.jumped = record.travel <= CLICK_TRAVEL && largest >= JUMP;
      entries.push(record);
      while (entries.length > MAX_ENTRIES) entries.shift();
      if (record.jumped) {
        console.warn(
          `[${EXT_NAME}] a gesture of ${record.travel}px moved a widget by ${largest}:`,
          record,
        );
      }
    }, 0);
  }, true);

  return true;
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Diagnostics", "Trace region editor gestures"],
      name: "Record region editor gestures",
      tooltip:
        "Keep the last 200 presses made on a rectangle or region editor, with where the pointer "
        + "went and what the widgets did either side. A gesture that barely moved the pointer but "
        + "moved a number a long way is warned about in the browser console as it happens. Read "
        + "them with WASRegionTrace.snaps() or WASRegionTrace.all(). Nothing is written to disk "
        + "and nothing leaves the browser.",
      type: "boolean",
      defaultValue: false,
    },
  ],

  async nodeCreated(node) {
    if (!enabled() || !NODES.includes(node.comfyClass ?? node.type)) return;
    // The editor is appended after the node is built, so the listener waits for it rather than
    // finding nothing and giving up.
    let tries = 0;
    const attach = () => {
      if (watch(node) || tries > 20) return;
      tries += 1;
      setTimeout(attach, 100);
    };
    attach();
  },
});

window.WASRegionTrace = {
  /** Every gesture recorded, oldest first. */
  all: () => entries.slice(),
  /** Only the gestures where a click-sized movement moved a widget a long way. */
  snaps: () => entries.filter((entry) => entry.jumped),
  /** Drop everything recorded so far. */
  clear: () => { entries.length = 0; },
};
