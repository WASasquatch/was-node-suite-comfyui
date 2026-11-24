/**
 * What resolution a surface draws at, and when that resolution changed.
 *
 * `surfaceRatio` folds the graph's zoom into the device pixel ratio and `watchSurfaceRatio`
 * reports a change. `contentRatio` leaves the zoom out, for buffers a preview computes over.
 */

const LOG_NAME = "WASNodeSuite.Resolution";

// One device pixel per layout pixel is the floor.
const MIN_RATIO = 1;

// Pixels one backing store may hold, at four bytes each.
const SURFACE_PIXEL_BUDGET = 2000000;

// The ratio is rounded up to this step.
const RATIO_STEP = 0.25;

// The error the measurement carries, in layout pixels, subtracted before the ceil.
const MEASUREMENT_SLACK = 0.5;

/**
 * The ratio for one element, or null when it cannot be measured.
 *
 * @param {HTMLElement} element - The interface root the canvas fills.
 * @returns {number|null} The ratio, or null for an element that is hidden, detached or unmeasured.
 */
function measureRatio(element) {
  const width = element?.offsetWidth ?? 0;
  const height = element?.offsetHeight ?? 0;
  if (!(width > 0) || !(height > 0)) return null;

  // The quotient of the two boxes is the scale the surface is drawn at.
  const drawn = Number(element.getBoundingClientRect?.().width);
  const scale = drawn > 0 ? drawn / width : 1;

  const density = Number(window.devicePixelRatio);
  const wanted = (density > 0 ? density : 1) * scale;

  // The budget is worked back from the element's own layout size.
  const affordable = Math.sqrt(SURFACE_PIXEL_BUDGET / (width * height));
  const slack = (wanted * MEASUREMENT_SLACK) / width;
  const stepped = Math.ceil((wanted - slack) / RATIO_STEP) * RATIO_STEP;
  return Math.max(MIN_RATIO, Math.min(stepped, affordable));
}

/**
 * The resolution one surface computes its content at, in pixels per layout pixel.
 *
 * @param {HTMLElement} element - The interface root the canvas fills.
 * @returns {number} A ratio of at least 1, never zero and never NaN.
 */
export function contentRatio(element) {
  const density = Number(window.devicePixelRatio);
  const wanted = density > 0 ? density : MIN_RATIO;

  // The same budget as a backing store's.
  const width = element?.offsetWidth ?? 0;
  const height = element?.offsetHeight ?? 0;
  const affordable = width > 0 && height > 0
    ? Math.sqrt(SURFACE_PIXEL_BUDGET / (width * height))
    : wanted;
  return Math.max(MIN_RATIO, Math.min(wanted, affordable));
}

/**
 * The resolution one surface draws at, in device pixels per layout pixel.
 *
 * @param {HTMLElement} element - The interface root the canvas fills.
 * @returns {number} A ratio of at least 1, never zero and never NaN.
 */
export function surfaceRatio(element) {
  return measureRatio(element) ?? MIN_RATIO;
}

// Every watched surface on the page, as `{element, onChange, ratio}`. A set of entries rather
// than a map keyed on the element, so a node carrying two interfaces on one element still gets
// two callbacks and each teardown removes its own.
const watched = new Set();

// The entries still waiting for a box, against the element each waits on, never read by the frame
// loop. A panel is built before it is shown and sometimes instead of ever being shown: copying a
// node clones it, a clone runs `onNodeCreated` and builds a panel, and the clone is never added
// to a graph, so the frontend never mounts its element and the `onRemoved` an interface hangs its
// teardown on never fires. Put in `watched` at construction, one entry per copied node measures 0
// on every frame for the life of the page and holds the loop open past the last interface
// anybody can see. The map is weak, so an element nothing else holds takes the entries waiting on
// it, and the callbacks they carry, with it.
const waiting = new WeakMap();

let frameHandle = 0;

// Frames between two full measurements while the zoom and the pixel density hold still.
const SWEEP_FRAMES = 60;

let viewSeen = "";

let framesQuiet = 0;

// One observer for the page, made by the first entry that has to wait and then kept: a panel that
// is never mounted is never disposed either, so there is no last waiter to give it back at, and
// an observer holding no targets is one object. A ResizeObserver reports a target the first time
// it is rendered at a size and reports nothing at all for one that is detached or hidden, which
// is what separates a panel on its way onto a node from a panel that will never be on one. It
// cannot replace the frame loop, since the graph's zoom is a transform and moves no border box.
let boxObserver = null;

/**
 * Tell one watcher its ratio moved.
 *
 * @param {object} entry - The watched surface.
 * @returns {void}
 */
function notify(entry) {
  try {
    entry.onChange();
  } catch (error) {
    console.error(`[${LOG_NAME}] A surface failed to follow a scale change:`, error);
  }
}

/**
 * The graph's zoom and the pixel density, as one string.
 *
 * @returns {string} Both values joined.
 */
function viewOf() {
  return `${window.devicePixelRatio}|${window.app?.canvas?.ds?.scale ?? ""}`;
}

/**
 * Measure every watched element when the zoom or the pixel density moved, and every
 * `SWEEP_FRAMES` frames otherwise, and tell the ones whose ratio moved.
 *
 * @returns {void}
 */
function tick() {
  frameHandle = 0;
  const view = viewOf();
  framesQuiet += 1;
  if (view === viewSeen && framesQuiet < SWEEP_FRAMES) {
    if (watched.size) frameHandle = requestAnimationFrame(tick);
    return;
  }
  viewSeen = view;
  framesQuiet = 0;
  for (const entry of watched) {
    const ratio = measureRatio(entry.element);
    // A hidden element measures zero, which is no answer rather than an answer of one. Keeping
    // the last one means the interface hears about the zoom it missed when it is shown again,
    // instead of a phantom change on every frame it spends collapsed.
    if (ratio === null || ratio === entry.ratio) continue;
    entry.ratio = ratio;
    notify(entry);
  }
  // A hidden tab runs no animation frame, so the loop idles there on its own and needs no
  // visibility listener to hold it back.
  if (watched.size) frameHandle = requestAnimationFrame(tick);
}

/**
 * Put one entry in the frame loop, starting the loop when it was stopped.
 *
 * @param {object} entry - The entry to watch.
 * @param {number|null} ratio - What its element measures now, or null when it has no box.
 * @returns {void}
 */
function join(entry, ratio) {
  entry.ratio = ratio;
  watched.add(entry);
  if (!frameHandle) frameHandle = requestAnimationFrame(tick);
}

/**
 * Take one entry out of the wait, and the element with it once nothing waits on it.
 *
 * @param {object} entry - The entry leaving the wait, by joining the loop or by teardown.
 * @returns {void}
 */
function stopWaiting(entry) {
  const holders = waiting.get(entry.element);
  // `unobserve` takes an element rather than one observation of it, so an element a second entry
  // is still waiting on stays observed for that entry.
  if (!holders?.delete(entry) || holders.size) return;
  waiting.delete(entry.element);
  boxObserver?.unobserve(entry.element);
}

/**
 * Take into the frame loop every element the observer reports that now has a box.
 *
 * @param {ResizeObserverEntry[]} reports - What the observer measured this delivery.
 * @returns {void}
 */
function admitWaiting(reports) {
  for (const report of reports) {
    const element = report?.target;
    const holders = element ? waiting.get(element) : null;
    if (!holders) continue;
    // An element is reported for any change in its box, so it can be reported while still
    // unmeasurable: a panel that has been given a width and no height yet stays where it is and
    // is reported again when it has both.
    const ratio = measureRatio(element);
    if (ratio === null) continue;
    waiting.delete(element);
    boxObserver?.unobserve(element);
    for (const entry of holders) {
      join(entry, ratio);
      // The surface has been drawing at the floor `surfaceRatio` answers for an element it
      // cannot measure, so the first real measurement is a change to report, exactly as the
      // frame loop reports the one it finds when a collapsed element comes back.
      notify(entry);
    }
  }
}

/**
 * Hold one entry out of the frame loop until its element is given a box.
 *
 * @param {object} entry - The entry to hold.
 * @returns {void}
 */
function waitForBox(entry) {
  if (typeof ResizeObserver !== "function") {
    // With no observer there is no signal to wait for, so the entry is watched from now and an
    // element that never gets a box is measured for the life of the page. Every browser the
    // frontend supports has one.
    join(entry, null);
    return;
  }
  const holders = waiting.get(entry.element);
  if (holders) {
    holders.add(entry);
    return;
  }
  waiting.set(entry.element, new Set([entry]));
  if (!boxObserver) boxObserver = new ResizeObserver(admitWaiting);
  boxObserver.observe(entry.element);
}

/**
 * Call back when an element's ratio changes.
 *
 * The callback runs only when the stepped ratio moved.
 *
 * @param {HTMLElement} element - The interface root the canvas fills.
 * @param {() => void} onChange - Called with no arguments, usually the interface's `schedulePaint`.
 * @returns {() => void} Teardown, which does nothing the second time it is called.
 */
export function watchSurfaceRatio(element, onChange) {
  if (!element || typeof onChange !== "function") return () => {};

  const entry = { element, onChange, ratio: null };
  const ratio = measureRatio(element);
  if (ratio === null) waitForBox(entry);
  else join(entry, ratio);

  let live = true;
  return () => {
    // Interface teardown runs from more than one path and some of them run twice, so the flag is
    // what makes the second call free rather than the set's own tolerance for it.
    if (!live) return;
    live = false;
    stopWaiting(entry);
    watched.delete(entry);
    if (!watched.size && frameHandle) {
      cancelAnimationFrame(frameHandle);
      frameHandle = 0;
    }
  };
}
