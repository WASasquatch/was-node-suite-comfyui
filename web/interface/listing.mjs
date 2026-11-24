/**
 * One body of a report drawn as a listing, one entry per line, paged as it is scrolled.
 *
 * `listingWindow` and `pagesNeeded` are the geometry behind it, in lines, rows and pixels.
 */

import { themeVar } from "./theme.mjs";

const LOG_NAME = "WASNodeSuite.Listing";

// A listing runs in columns from this many lines, while none is longer than this many
// characters.
const COLUMN_LINES = 8;
const COLUMN_CHARS = 44;

// Gaps between the cells of a listing, in CSS pixels.
const COLUMN_GAP = 14;
const ROW_GAP = 1;

// The style every cell of a listing carries.
const CELL_STYLE = "overflow-x:hidden;text-overflow:ellipsis;white-space:nowrap;min-height:1.35em";

// Rows drawn beyond each edge of the view, and released past.
const MARGIN_ROWS = 8;

// Rows drawn beyond each edge on the first pass, which is the one the layout is measured off.
const FIRST_ROWS = 64;

// The height of a row before the layout has been measured, in CSS pixels.
const ROW_GUESS = 16;

// Lines one request asks for.
const PAGE_LINES = 200;

// Pages held after the view has left them.
const CACHED_PAGES = 8;

// Milliseconds a view movement settles for before the pages it needs are asked for.
const SETTLE_MS = 60;

// Requests in flight at once.
const MAX_REQUESTS = 2;

// The height of the position chip before it has been drawn, in CSS pixels.
const CHIP_HEIGHT = 15;

// Overflow values that make an element the one a wheel scrolls.
const SCROLLS = ["auto", "scroll", "overlay"];

/**
 * The rows and lines a view covers, and the space to leave for the rest.
 *
 * @param {{top: number, height: number}} view - The visible box, in the listing's own pixels
 *   from its top edge.
 * @param {{columns: number, pitch: number, total: number}} layout - Cells across, pixels from
 *   one row to the next, and lines the whole body holds.
 * @param {number} [margin] - Rows drawn beyond each edge of the view.
 * @returns {{columns: number, pitch: number, rows: number, total: number, first: number,
 *   last: number, firstRow: number, lead: number, tail: number, seenFirst: number,
 *   seenLast: number}} The measured layout, the line range to draw, the row it opens on, the
 *   height of the spacer above and below it, and the first and last line in view, counted
 *   from one.
 */
export function listingWindow(view, layout, margin = MARGIN_ROWS) {
  const columns = Math.max(1, Math.trunc(Number(layout?.columns)) || 1);
  const pitch = Number(layout?.pitch) > 0 ? Number(layout.pitch) : ROW_GUESS;
  const total = Math.max(0, Math.trunc(Number(layout?.total)) || 0);
  const rows = Math.ceil(total / columns);
  const top = Math.max(0, Number(view?.top) || 0);
  const height = Math.max(0, Number(view?.height) || 0);
  const beyond = Math.max(0, Math.trunc(Number(margin)) || 0);
  const topRow = Math.min(rows, Math.floor(top / pitch));
  const lowRow = Math.min(rows, Math.ceil((top + height) / pitch));
  const firstRow = Math.max(0, topRow - beyond);
  const lastRow = Math.min(rows, lowRow + beyond);
  const first = firstRow * columns;
  const last = Math.min(total, lastRow * columns);
  const drawnRows = Math.ceil((last - first) / columns);
  const after = Math.max(0, rows - firstRow - drawnRows);
  const seenFirst = total ? Math.min(total, topRow * columns + 1) : 0;
  return {
    columns,
    pitch,
    rows,
    total,
    first,
    last,
    firstRow,
    lead: firstRow > 0 ? firstRow * pitch - ROW_GAP : 0,
    tail: after > 0 ? after * pitch - ROW_GAP : 0,
    seenFirst,
    seenLast: Math.max(seenFirst, Math.min(total, lowRow * columns)),
  };
}

/**
 * The pages a range of lines needs, in the order they are wanted.
 *
 * @param {number} first - The first line of the range, counting from zero.
 * @param {number} last - One line past the end of the range.
 * @param {number} have - Lines the report itself carried, which need no request.
 * @param {number} held - Lines still reachable, past which there is nothing to ask for.
 * @param {number} [size] - Lines a page carries.
 * @returns {number[]} Each page's index, counting from zero.
 */
export function pagesNeeded(first, last, have, held, size = PAGE_LINES) {
  const lines = Math.max(1, Math.trunc(Number(size)) || PAGE_LINES);
  const from = Math.max(0, Math.trunc(Number(first)) || 0, Math.trunc(Number(have)) || 0);
  const to = Math.min(Math.trunc(Number(last)) || 0, Math.trunc(Number(held)) || 0);
  const pages = [];
  if (from >= to) return pages;
  for (let page = Math.floor(from / lines); page * lines < to; page += 1) pages.push(page);
  return pages;
}

/**
 * One body's text, drawn as a listing of many short lines and as written otherwise.
 *
 * A listing holds only the rows in view.
 *
 * @param {string} text - The body's text, one entry per line.
 * @param {object} [options] - What the text is a piece of, and how to read the rest.
 * @param {number} [options.lines] - Lines the whole body holds. The lines in `text` by default.
 * @param {boolean} [options.whole] - False when `text` is a piece of a longer body.
 * @param {number} [options.offset] - Where `text` opens in the whole body, in characters.
 * @param {string} [options.name] - The body's name, matched against the name a page answers
 *   with so a page of another body is dropped.
 * @param {number} [options.run] - The run the report was published on, matched the same way.
 * @param {(start: number, count: number) => Promise<{name: string, lines: string[],
 *   total: number, held: number, run: number}|null>} [options.page] - Reads a range of lines.
 * @returns {HTMLElement} The element, carrying a `dispose` its owner calls when it is dropped.
 */
export function createListing(text, options = {}) {
  const carried = typeof text === "string" ? text : String(text ?? "");
  const read = typeof options.page === "function" ? options.page : null;
  // Whether the piece opens on the body's first character, which makes its lines the
  // body's first lines.
  const anchored = count(options.offset) === 0;
  const seed = carried.length ? carried.split("\n") : [];
  // The last line of a piece cut to a character count is dropped, and the reader answers
  // from there on in whole lines.
  const cut = options.whole === false && read && anchored && seed.length > 1;
  if (cut) seed.pop();
  const longest = seed.reduce((wide, line) => Math.max(wide, line.length), 0);
  if (seed.length <= COLUMN_LINES || longest > COLUMN_CHARS) return written(carried);

  const total = read && anchored
    ? Math.max(seed.length + (cut ? 1 : 0), count(options.lines))
    : seed.length;
  const name = typeof options.name === "string" ? options.name : "";
  const run = count(options.run);

  const listing = document.createElement("div");
  listing.dataset.wasListing = "grid";
  if (total > seed.length) listing.dataset.wasPaged = "1";
  let widest = longest;
  // Rows are their own content and are never squeezed to fit the room there is, so a line is
  // never cut across the middle. The entries run across the panel in as many columns as its
  // width holds instead of down one edge of it, and past the bottom the listing scrolls.
  listing.style.cssText = "display:grid;position:relative;min-height:0;overflow:auto;"
    + `align-content:start;column-gap:${COLUMN_GAP}px;row-gap:${ROW_GAP}px;`
    + `grid-auto-rows:min-content;color:${themeVar("fg")};`
    + `grid-template-columns:${columnsOf(widest)}`;

  const lead = spacer();
  const tail = spacer();
  const chip = position();
  listing.append(lead, tail, chip);

  const cells = [];
  const blanks = new Set();
  const cache = new Map();
  const asking = new Map();
  let columns = 1;
  let pitch = ROW_GUESS;
  let first = 0;
  let last = 0;
  let held = total;
  let width = 0;
  let chipHeight = 0;
  let measured = false;
  let opened = false;
  let disposed = false;
  let mounted = false;
  let scroller = null;
  let frame = 0;
  let timer = 0;

  /**
   * One line of the body, or null while it is not in hand.
   *
   * @param {number} index - The line, counting from zero.
   * @returns {string|null} The line.
   */
  const lineAt = (index) => {
    if (index < seed.length) return seed[index];
    const page = cache.get(Math.floor(index / PAGE_LINES));
    return page ? page[index % PAGE_LINES] ?? "" : null;
  };

  /**
   * Hold one page, dropping the page furthest from the view once the cache is full.
   *
   * @param {number} key - The page's index.
   * @param {string[]} lines - Its lines.
   * @returns {void}
   */
  const remember = (key, lines) => {
    cache.set(key, lines);
    while (cache.size > CACHED_PAGES) {
      const middle = (first + last) / 2 / PAGE_LINES;
      let furthest = key;
      let away = -1;
      for (const page of cache.keys()) {
        const gap = Math.abs(page - middle);
        if (gap > away) {
          away = gap;
          furthest = page;
        }
      }
      if (furthest === key) break;
      cache.delete(furthest);
    }
  };

  /**
   * One cell, carrying its line where that line is in hand.
   *
   * @param {number} index - The line the cell draws.
   * @returns {HTMLElement} The cell.
   */
  const cellFor = (index) => {
    const cell = document.createElement("div");
    cell.style.cssText = CELL_STYLE;
    const said = lineAt(index);
    if (said === null) blanks.add(index);
    else cell.textContent = said;
    return cell;
  };

  /**
   * Hold the drawn cells to one range of lines, building and releasing at both ends.
   *
   * @param {number} from - The first line to draw.
   * @param {number} upto - One line past the last.
   * @returns {void}
   */
  const place = (from, upto) => {
    const to = Math.max(from, upto);
    if (!cells.length || from >= last || to <= first) {
      for (const cell of cells) cell.remove();
      cells.length = 0;
      blanks.clear();
      first = from;
      last = from;
    }
    while (first < from) {
      cells.shift().remove();
      blanks.delete(first);
      first += 1;
    }
    while (last > to) {
      cells.pop().remove();
      blanks.delete(last - 1);
      last -= 1;
    }
    while (first > from) {
      first -= 1;
      const cell = cellFor(first);
      cells.unshift(cell);
      listing.insertBefore(cell, cells[1] ?? tail);
    }
    while (last < to) {
      const cell = cellFor(last);
      cells.push(cell);
      listing.insertBefore(cell, tail);
      last += 1;
    }
  };

  /**
   * The part of the listing the scroller shows, in the listing's own pixels.
   *
   * @returns {{top: number, height: number}} The top edge and the height of the view.
   */
  const view = () => {
    const room = listing.clientHeight;
    // Squeezed into a panel shorter than its rows, the listing is what scrolls.
    if (listing.scrollHeight > room + 1) return { top: listing.scrollTop, height: room };
    const box = listing.getBoundingClientRect();
    const laid = listing.offsetHeight;
    // The painted height over the laid out height, which a zoomed node scales apart.
    const scale = laid > 0 && box.height > 0 ? box.height / laid : 1;
    const clip = scroller ? scroller.getBoundingClientRect() : null;
    const top = Math.max(0, ((clip ? clip.top : 0) - box.top) / scale);
    const bottom = Math.min(laid, ((clip ? clip.bottom : windowHeight()) - box.top) / scale);
    return { top, height: Math.max(0, bottom - top) };
  };

  /**
   * Read the column count and the row height off the cells the last pass drew.
   *
   * @returns {boolean} True when either moved.
   */
  const measure = () => {
    if (!cells.length) return false;
    const top = cells[0].offsetTop;
    let across = 1;
    let step = 0;
    for (let index = 1; index < cells.length; index += 1) {
      const at = cells[index].offsetTop;
      if (at === top) {
        across += 1;
        continue;
      }
      step = at - top;
      break;
    }
    const before = `${columns}:${pitch}`;
    const height = cells[0].offsetHeight;
    if (step > 0) {
      columns = across;
      pitch = step;
      measured = true;
    } else if (height > 0) {
      columns = across;
      pitch = height + ROW_GAP;
      measured = true;
    }
    opened = opened || measured;
    return `${columns}:${pitch}` !== before;
  };

  /**
   * Draw the rows the view covers and leave the room the rest take.
   *
   * @returns {object} The window `listingWindow` answered.
   */
  const draw = () => {
    const seen = view();
    const shape = listingWindow(seen, { columns, pitch, total },
      opened ? MARGIN_ROWS : FIRST_ROWS);
    place(shape.first, shape.last);
    space(lead, shape.lead);
    space(tail, shape.tail);
    mark(shape, seen);
    return shape;
  };

  /**
   * Draw where the view sits in the body, over the bottom of the view itself.
   *
   * @param {object} shape - The window `listingWindow` answered.
   * @param {{top: number, height: number}} seen - The view that window was worked out from.
   * @returns {void}
   */
  const mark = (shape, seen) => {
    const more = shape.rows * pitch > seen.height + 1;
    chip.style.display = more ? "block" : "none";
    if (!more) return;
    const missing = Math.max(0, total - held);
    chip.textContent = `${shape.seenFirst} to ${shape.seenLast} of ${total}`
      + (missing > 0 ? `, ${missing} no longer held` : "");
    chipHeight = chipHeight || chip.offsetHeight || CHIP_HEIGHT;
    chip.style.top = `${Math.max(0, seen.top + seen.height - chipHeight - 2)}px`;
  };

  /**
   * Draw again, once, on the next frame.
   *
   * @returns {void}
   */
  const soon = () => {
    if (disposed || frame) return;
    frame = requestAnimationFrame(render);
  };

  /**
   * Draw the view, measuring the layout again where it has moved.
   *
   * @returns {void}
   */
  const render = () => {
    frame = 0;
    if (disposed || !listing.isConnected) return;
    // The width is read before anything this frame is written.
    const now = listing.clientWidth;
    const reflowed = now !== width;
    width = now;
    let shape = draw();
    if ((!measured || reflowed) && measure()) shape = draw();
    settle(shape);
  };

  /**
   * Ask for the pages the window needs once the view has stopped moving.
   *
   * @param {object} shape - The window `listingWindow` answered.
   * @returns {void}
   */
  const settle = (shape) => {
    if (!read || disposed) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = 0;
      ask(shape);
    }, SETTLE_MS);
  };

  /**
   * Start the requests one window needs, up to the number allowed at once.
   *
   * @param {object} shape - The window `listingWindow` answered.
   * @returns {void}
   */
  const ask = (shape) => {
    if (disposed || !listing.isConnected) return;
    for (const page of pagesNeeded(shape.first, shape.last, seed.length, held)) {
      if (cache.has(page) || asking.has(page)) continue;
      if (asking.size >= MAX_REQUESTS) return;
      fetchPage(page);
    }
  };

  /**
   * Read one page and hold it.
   *
   * @param {number} page - The page's index.
   * @returns {void}
   */
  const fetchPage = (page) => {
    const start = page * PAGE_LINES;
    asking.set(page, Promise.resolve()
      .then(() => read(start, PAGE_LINES))
      .then((answer) => took(page, answer))
      .catch((error) => {
        console.error(`[${LOG_NAME}] Failed to read the lines from ${start}:`, error);
      })
      .finally(() => asking.delete(page)));
  };

  /**
   * Take one page in, drawing it where the view still covers it.
   *
   * @param {number} page - The page's index.
   * @param {object|null} answer - What the reader answered.
   * @returns {void}
   */
  const took = (page, answer) => {
    if (disposed || !listing.isConnected || !answer || !Array.isArray(answer.lines)) return;
    // Pages of another body and of another run are dropped.
    if (name && answer.name && answer.name !== name) return;
    if (run && answer.run && answer.run !== run) return;
    const before = held;
    if (Number.isFinite(Number(answer.held))) held = Math.min(total, count(answer.held));
    remember(page, answer.lines);
    grow(answer.lines);
    const from = page * PAGE_LINES;
    // A page the view has left is held for the return rather than drawn.
    const covered = from < last && from + answer.lines.length > first;
    if (covered) {
      for (const index of Array.from(blanks)) {
        const said = lineAt(index);
        if (said === null) continue;
        cells[index - first].textContent = said;
        blanks.delete(index);
      }
    }
    if (covered || held !== before) soon();
  };

  /**
   * Widen the columns for a line longer than any drawn so far.
   *
   * @param {string[]} lines - The lines that arrived.
   * @returns {void}
   */
  const grow = (lines) => {
    let widened = widest;
    for (const line of lines) widened = Math.max(widened, line.length);
    widened = Math.min(widened, COLUMN_CHARS);
    if (widened === widest) return;
    widest = widened;
    listing.style.gridTemplateColumns = columnsOf(widest);
    measured = false;
  };

  const onScroll = () => soon();

  const observer = new ResizeObserver(() => {
    if (disposed) return;
    if (!listing.isConnected) {
      observer.disconnect();
      return;
    }
    if (!mounted) {
      mounted = true;
      scroller = scrollParent(listing);
      listing.addEventListener("scroll", onScroll, { passive: true });
      (scroller ?? window).addEventListener("scroll", onScroll, { passive: true });
      if (scroller) observer.observe(scroller);
    }
    soon();
  });
  observer.observe(listing);

  listing.dispose = () => {
    if (disposed) return;
    disposed = true;
    observer.disconnect();
    listing.removeEventListener("scroll", onScroll);
    (scroller ?? window).removeEventListener("scroll", onScroll);
    if (frame) cancelAnimationFrame(frame);
    if (timer) clearTimeout(timer);
  };

  draw();
  return listing;
}

/**
 * The track list a listing lays its columns out on.
 *
 * @param {number} chars - Characters of the longest entry in hand.
 * @returns {string} The `grid-template-columns` value.
 */
function columnsOf(chars) {
  // `min(100%, ...)` holds a track to the panel's own width.
  return `repeat(auto-fill,minmax(min(100%,${chars + 2}ch),1fr))`;
}

/**
 * A full width row standing in for the rows outside the window.
 *
 * @returns {HTMLElement} The spacer, drawn only once it is given a height.
 */
function spacer() {
  const gap = document.createElement("div");
  gap.style.cssText = "grid-column:1/-1;display:none";
  return gap;
}

/**
 * Give one spacer its height, or take it out of the grid.
 *
 * @param {HTMLElement} gap - The spacer.
 * @param {number} height - Pixels it stands for.
 * @returns {void}
 */
function space(gap, height) {
  const room = Math.max(0, Math.round(height));
  gap.style.display = room > 0 ? "block" : "none";
  if (room > 0) gap.style.height = `${room}px`;
}

/**
 * The chip naming the lines in view.
 *
 * @returns {HTMLElement} The chip, drawn over the bottom of the view.
 */
function position() {
  const chip = document.createElement("div");
  chip.style.cssText = "position:absolute;right:2px;display:none;pointer-events:none;"
    + "padding:0 5px;border-radius:3px;font-size:9px;line-height:1.6;white-space:nowrap;"
    + `background:${themeVar("panelBg")};color:${themeVar("fgMuted")};`
    + `border:1px solid ${themeVar("border")}`;
  return chip;
}

/**
 * A body drawn as it was written.
 *
 * @param {string} text - The body's text.
 * @returns {HTMLElement} The block, wrapped rather than cut, so a long line stays read.
 */
function written(text) {
  const block = document.createElement("pre");
  block.dataset.wasListing = "text";
  block.style.cssText = "margin:0;white-space:pre-wrap;word-break:break-word;"
    + `overflow:auto;min-height:0;color:${themeVar("fg")}`;
  block.textContent = text;
  return block;
}

/**
 * The element a listing scrolls inside.
 *
 * @param {HTMLElement} element - The listing.
 * @returns {HTMLElement|null} The nearest ancestor that scrolls, or null for the page itself.
 */
function scrollParent(element) {
  for (let above = element.parentElement; above; above = above.parentElement) {
    const style = getComputedStyle(above);
    if (SCROLLS.includes(style.overflowY) || SCROLLS.includes(style.overflow)) return above;
    if (above === document.body || above === document.documentElement) break;
  }
  return null;
}

/**
 * The height of the page's own view.
 *
 * @returns {number} The height in CSS pixels.
 */
function windowHeight() {
  return window.innerHeight || document.documentElement?.clientHeight || 0;
}

/**
 * A value as a whole number of at least zero.
 *
 * @param {*} value - Whatever the caller handed over.
 * @returns {number} The number, 0 for anything that is not one.
 */
function count(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.trunc(number)) : 0;
}
