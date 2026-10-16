/**
 * Run readout for the Text Find and Replace node.
 *
 * Draws the match and change counts, one total per pattern, and the text as searched and as it
 * came out with every span marked.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { growWidgets } from "./interface/grow.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel } from "./interface/pointer.js";
import { PREVIEW_STATE } from "./interface/preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeDone } from "./interface/run_events.js";
import {
  RUN_LABELS,
  RUN_STATUS,
  TRUNCATED,
  drawableBody,
  fetchRunResult,
  sameValue,
  visibleText,
  wrapBody,
} from "./interface/run_result.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.SearchAndReplaceUI";
const NODE_NAME = "Text Find and Replace";
const SETTING_ID = "WAS.SearchAndReplace.ShowInterface";
const HEIGHT_SETTING_ID = "WAS.SearchAndReplace.PanelHeight";

const UI_WIDGET_NAME = "was_search_and_replace_ui";
const UI_WIDGET_TYPE = "was_run_readout";

// The three inputs that decide what a run does, in the order the schema declares them. Each can
// be filled by a link, and a link is read instead of the box beside it.
const INPUT_NAMES = ["text", "find", "replace"];

// The count naming the matches whose replacement differs from the text they matched. The node
// leaves it out of a run with more matches than it walks, which is the one number this panel
// can be missing.
const REPLACED_COUNT = "replaced";

// How long an answer stands before the pointer arriving over the panel asks again, which is
// what covers a run whose end event was missed.
const STALE_MS = 3000;

const UI_MARGIN = 10;
const DEFAULT_HEIGHT = 260;
const MIN_HEIGHT = 148;
const MAX_HEIGHT = 900;


// Layout bands, in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const COUNTS_HEIGHT = 22;
const STATE_HEIGHT = 14;
const ROW_HEIGHT = 14;
const FOOTER_HEIGHT = 13;
const BAND_GAP = 3;
const GLYPH_GAP = 4;
const CELL_GAP = 12;
const LABEL_GAP = 3;
const NOTE_GAP = 6;
const ROW_PAD = 3;

// A block of text: the line naming it, one line of its text, and the padding inside its box.
const BLOCK_LABEL_HEIGHT = 12;
const BLOCK_LINE_HEIGHT = 12;
const BLOCK_PAD = 3;

// How much of the body the sample rows may take when there are both blocks and rows, so the
// blocks keep the room they are the point of.
const ROWS_SHARE = 0.4;

// How much of the body the per pattern totals may take, so eight pairs cannot leave the blocks
// below them nothing. The band draws the rows that fit and says how many it drew.
const PATTERNS_SHARE = 0.45;

// What the band of per pattern totals is called on the node.
const PATTERNS_LABEL = "patterns";

// How a marked span is drawn: a wash of the accent behind the characters, a rule of it at full
// strength under them, and the characters themselves at full contrast while everything around
// them is muted. Three cues rather than one, so a mark reads where a wash alone would not.
const MARK_ALPHA = 0.28;
const MARK_RULE = 1;

// Width of the bar drawn for a span of no width, which marks a position rather than characters.
// Drawn at full strength, since two pixels of a wash mark nothing anybody can see.
const CARET_WIDTH = 2;

// The dashes of a box edge the text carries on past.
const EDGE_DASH = [2, 3];

// Room a row keeps for the excerpt before the note beside it is given up, and how much of the
// match is held clear of the right edge when a row has to be shifted to reach it.
const MIN_TEXT_WIDTH = 72;
const MARK_INSET = 10;

const FIGURE_FONT = "13px sans-serif";
const BODY_FONT = "10px sans-serif";
const LABEL_FONT = "9px sans-serif";

// What the band of numbers says on hover, which is the one thing about them that never changes.
const COUNTS_HOVER = "found counts every match the pattern made. replaced counts the matches "
  + "whose replacement differs from the text they matched, so a match replaced with what it "
  + "already said is counted as found and not as replaced.";

// What the footer says on hover. The line above it names the input a value arrived on, and that
// is the whole of what separates a run that read the box from one that read the wire.
const FOOTER_HOVER = "The run read what this line names. Where a link is wired into an input, "
  + "the box on the node is not read at all.";

// What the empty panel says on hover. A node that has run can still have nothing to report.
const WAITING_HOVER = "A node reports its run while a browser is attached to the server, and "
  + "the report is held in memory for as long as ComfyUI runs. Queue the graph and the numbers "
  + "arrive here.";

// What the two blocks say on hover. Where they hold the whole text this is all there is to say
// about them; a block holding a window says so on its own line as well.
const BLOCKS_HOVER = "before is the text the run searched and after is the text it produced, "
  + "each with every span the run found or wrote marked in place. Line breaks are the text's "
  + "own, and a line too long for the box is wrapped rather than cut off.";

// What the per pattern totals say on hover. The rows themselves name each pair and its number;
// this is what a number of zero means and why the numbers add up to the total found above.
const PATTERNS_HOVER = "Every pattern the run applied, with how many matches each of them made. "
  + "A pattern at zero matched nothing, so its box changed the text in no way. The totals add up "
  + "to found above, since a match is made by one pattern: where two patterns match in the same "
  + "place the earlier box wins, and the later one does not count it.";

// What a dashed edge of a block means. The edge itself is the state and is drawn; this is why
// it is there.
const EDGE_HOVER = "A dashed edge of the box means the text carries on past it. A solid edge is "
  + "the start or the end of the text itself.";

// What the sample rows say on hover, which is the one thing about them a reader has to know:
// they are not a second copy of what the blocks already show.
const ROWS_HOVER = "Matches too far into the text for the blocks above to reach, each in "
  + "context with the line it sits on. No two of them draw the same passage, and a line break "
  + "inside one reads as a space, since a row is one line.";

// The same list where there are no blocks to compare it against.
const ROWS_ONLY_HOVER = "The first matches in context, each with the line it sits on. A line "
  + "break inside one reads as a space, since a row is one line.";

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
 * Whether one input still holds the value the run was handed on it.
 *
 * @param {object} entry - One entry of the report's `inputs`.
 * @param {object} node - The node the panel is drawn on.
 * @param {Map<string, object>} memo - What each box held when it was last measured.
 * @param {number} run - Which run the report is, so a new report is measured again.
 * @returns {boolean} True while the box holds what the run read.
 */
function holdsGiven(entry, node, memo, run) {
  const value = findWidget(node, entry.name)?.value;
  const said = typeof value === "string" ? value : String(value ?? "");
  const last = memo.get(entry.name);
  if (last && last.run === run && last.said === said) return last.held;
  const held = sameValue(entry, said);
  memo.set(entry.name, { run, said, held });
  return held;
}

/**
 * Which inputs are not what the run was handed, and which of them cannot be told.
 *
 * @param {object} report - The report on screen.
 * @param {object} node - The node the panel is drawn on.
 * @param {Map<string, object>} memo - What each box held when it was last measured.
 * @returns {{moved: string[], unknown: string[], held: string[]}} Names in the order the schema
 *   declares them: the ones that are not what the run read, the ones nothing here can answer
 *   for, and the ones measured as still holding it.
 */
function readGiven(report, node, memo) {
  const moved = [];
  const unknown = [];
  const held = [];
  for (const name of INPUT_NAMES) {
    const entry = (report.inputs ?? []).find((one) => one.name === name);
    const linked = inputLinked(node, name);
    // A report that does not name an input says nothing about it, which is what a run whose
    // prompt the server could not read publishes.
    if (!entry) {
      unknown.push(name);
      continue;
    }
    // The box beside a linked input was read by nothing on the run, so the link going away is
    // the one change on this node that bears on those numbers.
    if (entry.linked === true) {
      if (!linked) moved.push(name);
      continue;
    }
    // A link attached since the run: what the input carries now comes from upstream rather than
    // from the box the run read.
    if (linked) {
      moved.push(name);
      continue;
    }
    if (holdsGiven(entry, node, memo, report.run)) {
      held.push(name);
    } else if (entry.linked === false) {
      moved.push(name);
    } else {
      // A run that could not say whether it read the box leaves a box that differs from it
      // unaccounted for: it may be an edit, and it may be a value the run never read.
      unknown.push(name);
    }
  }
  return { moved, unknown, held };
}

/**
 * Name several things in one phrase.
 *
 * @param {string[]} names - What to name, already in the order to read them.
 * @returns {string} The names joined with commas and a final `and`.
 */
function listNames(names) {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/**
 * One number as it is drawn.
 *
 * @param {number} value - The count.
 * @returns {string} Its digits, grouped in threes the way the node's own summary groups them.
 */
function figure(value) {
  if (!Number.isFinite(value)) return "";
  if (!Number.isInteger(value)) return String(value);
  return String(value).replace(/\B(?=(\d{3})+$)/g, ",");
}

/**
 * Read whether the panel is drawn at all.
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
 * Read how tall the panel is drawn.
 *
 * @returns {number} The height in node units, between `MIN_HEIGHT` and `MAX_HEIGHT`.
 */
function panelHeight() {
  let wanted = DEFAULT_HEIGHT;
  try {
    const value = app?.extensionManager?.setting?.get?.(HEIGHT_SETTING_ID);
    const legacy = app?.ui?.settings?.getSettingValue?.(HEIGHT_SETTING_ID, DEFAULT_HEIGHT);
    const said = Number.isFinite(Number(value)) ? Number(value) : Number(legacy);
    if (Number.isFinite(said)) wanted = said;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${HEIGHT_SETTING_ID}:`, error);
  }
  return Math.max(MIN_HEIGHT, Math.min(Math.round(wanted), MAX_HEIGHT));
}

/**
 * What one block says about the text it holds beside the text itself.
 *
 * @param {object} body - One body of a report.
 * @returns {{brief: string, full: string}} How much of the text is marked on its own, and that
 *   with how much of the text is on screen, written so a window cannot be read as the file.
 */
function blockNote(body) {
  const brief = body.marksTotal === body.marks.length
    ? `${figure(body.marks.length)} marked`
    : `${figure(body.marks.length)} of ${figure(body.marksTotal)} marked`;
  if (body.whole) {
    return { brief, full: `${brief}, whole text, ${figure(body.length)} ${chars(body.length)}` };
  }
  const first = body.offset + 1;
  const last = body.offset + body.text.length;
  return {
    brief,
    full: `${brief}, characters ${figure(first)} to ${figure(last)} of ${figure(body.length)}`,
  };
}

/**
 * One line of text held inside the room it has, with an ellipsis in place of what was cut.
 *
 * @param {CanvasRenderingContext2D} ctx - Target context, carrying the font it is drawn in.
 * @param {string} said - The whole line.
 * @param {number} room - Pixels it may take.
 * @returns {string} The line, or as much of it as fits with `...` after it.
 */
function clipText(ctx, said, room) {
  if (room <= 0) return "";
  if (ctx.measureText(said).width <= room) return said;
  // Halved rather than stepped, so a line cut short costs a measurement per doubling of its
  // length rather than one per character.
  let low = 0;
  let high = said.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (ctx.measureText(`${said.slice(0, middle)}...`).width <= room) low = middle;
    else high = middle - 1;
  }
  return low > 0 ? `${said.slice(0, low)}...` : "";
}

/**
 * The longest form of a block's note that fits the room beside its name.
 *
 * @param {CanvasRenderingContext2D} ctx - Target context, carrying the label font.
 * @param {{brief: string, full: string}} note - What the block has to say.
 * @param {number} room - Pixels left of it.
 * @returns {string} The note to draw, empty when not even the short form fits.
 */
function fitNote(ctx, note, room) {
  for (const said of [note.full, note.brief]) {
    if (ctx.measureText(said).width <= room) return said;
  }
  return "";
}

/**
 * Which of a block's marks reach one line, in that line's own indices.
 *
 * @param {number[][]} marks - Every marked span of the block, in its text's indices.
 * @param {number} start - Where the line starts in that text.
 * @param {number} stop - One past its last character.
 * @param {number} next - Where the line drawn below it starts, or -1 for none.
 * @returns {{spans: number[][], carets: number[]}} Spans of characters, merged and in order,
 *   and positions between characters.
 */
function lineMarks(marks, start, stop, next) {
  const spans = [];
  const carets = [];
  for (const [first, last] of marks) {
    if (first === last) {
      // A wrapped line begins where the one above it ended, so a position sitting on the break
      // is drawn on the line that carries on rather than on both of them.
      if (first < start || first > stop || (first === stop && first === next)) continue;
      carets.push(first - start);
      continue;
    }
    // Touching an edge is not reaching the line: a span ending where this line begins belongs
    // to the line above, and drawing it here would put a mark of no width on the wrong text.
    if (first >= stop || last <= start) continue;
    spans.push([Math.max(first - start, 0), Math.min(last - start, stop - start)]);
  }
  spans.sort((one, two) => one[0] - two[0]);
  const merged = [];
  for (const span of spans) {
    const above = merged[merged.length - 1];
    if (above && span[0] <= above[1]) above[1] = Math.max(above[1], span[1]);
    else merged.push(span);
  }
  return { spans: merged, carets };
}

/**
 * Where the marked boundaries of one line sit, measured from its left edge.
 *
 * @param {CanvasRenderingContext2D} ctx - Target context, carrying the block's font.
 * @param {string} line - The line's text.
 * @param {number[][]} spans - Its spans of characters.
 * @param {number[]} carets - Its positions between characters.
 * @returns {Map<number, number>} Each boundary against how far into the line it sits.
 */
function markOffsets(ctx, line, spans, carets) {
  const cuts = [...new Set([...spans.flat(), ...carets, line.length])]
    .filter((cut) => cut > 0)
    .sort((one, two) => one - two);
  const at = new Map([[0, 0]]);
  let x = 0;
  let cursor = 0;
  for (const cut of cuts) {
    x += ctx.measureText(line.slice(cursor, cut)).width;
    at.set(cut, x);
    cursor = cut;
  }
  return at;
}

/**
 * Draw one horizontal edge of a box as dashes.
 *
 * @param {CanvasRenderingContext2D} ctx - Target context.
 * @param {object} theme - Tokens from `readTheme`.
 * @param {number} x0 - Left end.
 * @param {number} x1 - Right end.
 * @param {number} y - Where the edge sits.
 * @returns {void}
 */
function dashEdge(ctx, theme, x0, x1, y) {
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.fgMuted;
  ctx.setLineDash(EDGE_DASH);
  ctx.beginPath();
  ctx.moveTo(x0, y);
  ctx.lineTo(x1, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * The word for a number of characters.
 *
 * @param {number} count - How many.
 * @returns {string} `character` or `characters`.
 */
function chars(count) {
  return count === 1 ? "character" : "characters";
}

/**
 * Work out where each band of the panel sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the counts band, the body between it and the footer, and
 *   the footer.
 */
function computeLayout(width, height) {
  const x0 = PAD_X;
  const x1 = Math.max(x0 + 1, width - PAD_X);
  const countsY = PAD_Y;
  const bodyY = countsY + COUNTS_HEIGHT + BAND_GAP;
  const footerY = Math.max(bodyY + ROW_HEIGHT, height - PAD_Y - FOOTER_HEIGHT);
  return {
    width,
    height,
    x0,
    x1,
    countsY,
    countsHeight: COUNTS_HEIGHT,
    bodyY,
    bodyHeight: Math.max(ROW_HEIGHT, footerY - bodyY - BAND_GAP),
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * Build the run readout for one node.
 *
 * @param {object} node - The node the panel decorates.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   handleConfigured: () => void, dispose: () => void}} The panel for `appendInterfaceWidget`,
 *   a repaint, the reload a workflow load needs, and teardown.
 */
function createRunReadout(node) {
  const height = panelHeight();
  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${height - UI_MARGIN * 2}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The glyph, the rows and the footer state what they are worth through the element's own
  // title. The regions are handed over again on every repaint, since they move whenever the
  // node is resized.
  const titles = hoverTitles(root);

  const state = {
    // The report on screen and the run number it carries. A report names the values the run was
    // handed, so what the numbers are worth is measured against the report itself rather than
    // against anything this panel had to remember.
    report: null,
    run: 0,
    // What each box held when it was last measured against the report, keyed by input name.
    measured: new Map(),
    status: PREVIEW_STATE.WAITING,
    token: 0,
    fetchedAt: 0,
    // Page-wide listeners are taken at the moment the element is first given a box. Copying a
    // node builds a panel that is never mounted and never removed, and one joined at
    // construction would hold its listeners for the life of the page.
    mounted: false,
    unwatchRuns: null,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Ask the route what the node's last run did.
   *
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function load() {
    if (state.disposed) return;
    const token = (state.token += 1);
    if (!state.report) state.status = PREVIEW_STATE.LOADING;

    try {
      const answer = await fetchRunResult(node);
      if (state.disposed || token !== state.token) return;
      if (answer.result && answer.result.run !== state.run) {
        state.report = answer.result;
        state.run = answer.result.run;
      }
      // A report already drawn is kept through a refusal and through the 404 a node whose
      // report was evicted answers, since what it says about its run stayed true.
      state.status = state.report ? PREVIEW_STATE.READY : answer.state;
    } finally {
      if (!state.disposed) {
        state.fetchedAt = Date.now();
        schedulePaint();
      }
    }
  }

  /**
   * Ask for the report, logging a failure rather than throwing it at the caller.
   *
   * @returns {void}
   */
  function refresh() {
    load().catch((error) => {
      console.error(`[${EXT_NAME}] Failed to ask what the last run did:`, error);
    });
  }

  /**
   * Take the page-wide listeners, which happens once the element is really on a node.
   *
   * @returns {void}
   */
  function joinRuns() {
    if (state.unwatchRuns || typeof api?.addEventListener !== "function") return;
    // Asked for as soon as the node finishes, and again when a run ends without it finishing.
    const stopWatchingRuns = onNodeDone(node, () => refresh());
    // The socket opening is what turns the connecting state into an answer, and a status
    // arrives with it. Nothing is asked for while a report is already drawn.
    const onStatus = () => {
      if (state.status !== PREVIEW_STATE.READY) refresh();
    };
    api.addEventListener("status", onStatus);
    state.unwatchRuns = () => {
      if (typeof stopWatchingRuns === "function") stopWatchingRuns();
      api.removeEventListener?.("status", onStatus);
    };
  }

  /**
   * What the panel draws, worked out from the report and the widgets as they stand.
   *
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {object} The report, which inputs are no longer what the run read, the per pattern
   *   totals, the rows to draw, the line above them, the claim the glyph carries, and the words
   *   drawn in place of the whole readout when there is no report.
   */
  function readModel(theme) {
    const report = state.report;
    if (!report) {
      return {
        report: null,
        moved: [],
        rows: [],
        blocks: [],
        tallies: [],
        talliesTotal: 0,
        notice: RUN_LABELS[state.status] ?? "",
        noticeTitle: state.status === PREVIEW_STATE.WAITING ? WAITING_HOVER : "",
        claim: null,
      };
    }

    const given = readGiven(report, node, state.measured);
    const rows = report.items.map((row) => {
      const drawn = visibleText(row.text, row.mark);
      return { text: drawn.text, mark: drawn.mark, note: row.note };
    });
    const blocks = (report.bodies ?? []).map((body) => {
      const note = blockNote(body);
      return {
        name: body.name,
        text: drawableBody(body.text),
        marks: body.marks,
        // Whether the piece the body carries reaches each end of its text, which is what the
        // box's own top and bottom edges are drawn from.
        opens: body.offset === 0,
        ends: body.offset + body.text.length >= body.length,
        note,
        title: `${body.name}: ${note.full}\n${BLOCKS_HOVER}`,
      };
    });

    // One row per pattern the run applied, each drawn on one line, so a pattern holding a line
    // break reads as the pattern rather than as a break in the band.
    const tallies = (report.tallies ?? []).map((entry) => ({
      name: visibleText(entry.name, null).text,
      value: entry.value,
    }));

    let line = "";
    let tone = theme.warning;
    if (given.moved.length) {
      line = `${listNames(given.moved)} changed since this run`;
    } else if (report.status !== RUN_STATUS.OK) {
      line = report.summary;
      if (report.status === RUN_STATUS.ERROR) tone = theme.error;
    }

    return {
      report,
      moved: given.moved,
      rows,
      blocks,
      tallies,
      talliesTotal: report.talliesTotal,
      line,
      tone,
      // The summary is the node's own sentence about the run. It is drawn where it is the state
      // somebody has to act on, and where something more urgent took the line it goes on hover.
      lineTitle: line === report.summary ? "" : report.summary,
      notice: "",
      noticeTitle: "",
      claim: readClaim(given),
    };
  }

  /**
   * What the readout is worth against the run, as a glyph and the measurement behind it.
   *
   * @param {object} given - What `readGiven` answered.
   * @returns {{icon: string, detail: string}} The claim for `iconTitle`.
   */
  function readClaim(given) {
    const linked = INPUT_NAMES.filter((name) => inputLinked(node, name));
    // A link beats every other caveat about the future: the value arrives from upstream and can
    // change with nothing on this node to show it, which no edit here would report.
    const aside = linked.length
      ? `. ${listNames(linked)} ${linked.length > 1 ? "are" : "is"} filled by a link, which can `
        + "change upstream with nothing on this node to show it"
      : "";

    if (given.moved.length) {
      return {
        icon: ICON.WARNING,
        detail: `${listNames(given.moved)} ${given.moved.length > 1 ? "are" : "is"} not what `
          + "the run was handed, so these numbers describe the text as the node held it then "
          + `rather than as it stands now${aside}`,
      };
    }
    if (given.unknown.length) {
      return {
        icon: ICON.WARNING,
        detail: `the run did not report what it read on ${listNames(given.unknown)}, so whether `
          + `the node still holds what produced these numbers is not known here${aside}`,
      };
    }
    const missing = shortfall();
    if (missing) return { icon: ICON.APPROXIMATE, detail: `${missing}${aside}` };
    // The run published what it was handed on each input, so this is a comparison against the
    // run's own account of itself rather than a claim about when the values were read.
    const still = given.held.length
      ? `, and ${listNames(given.held)} ${given.held.length > 1 ? "are" : "is"} what the run `
        + "reported reading"
      : "";
    return {
      icon: ICON.EXACT,
      detail: "every number here was measured by the run itself, inside the node, rather than "
        + `worked out again in the browser${still}${aside}`,
    };
  }

  /**
   * What the run measured that is not on screen.
   *
   * @returns {string} The words for it, empty when the whole of what was measured is drawn.
   */
  function shortfall() {
    const report = state.report;
    const missing = [];
    if (!report.counts.some((count) => count.name === REPLACED_COUNT)) {
      missing.push("how many of the matches changed the text was not counted, since the run "
        + "made more of them than the node walks");
    }
    if (report.truncated.includes(TRUNCATED.SUMMARY)) missing.push("the summary line was cut");
    if (report.truncated.includes(TRUNCATED.TEXT)) missing.push("a row was cut to fit");
    if (report.truncated.includes(TRUNCATED.COUNTS)) missing.push("a count was left out");
    if (report.truncated.includes(TRUNCATED.FACTS)) missing.push("a fact was left out");
    if (report.truncated.includes(TRUNCATED.BODIES)) missing.push("a block was left out");
    if (report.truncated.includes(TRUNCATED.INPUTS)) {
      missing.push("what the run read on one of the inputs was left out");
    }
    return missing.join(", and ");
  }

  /**
   * Draw the numbers.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawCounts(ctx, theme, model, regions) {
    const layout = state.layout;
    const baseline = layout.countsY + COUNTS_HEIGHT - 5;
    const available = layout.x1 - layout.x0;
    // A moved input costs the numbers their weight rather than their place: they are still what
    // the run measured, and they are no longer what the node holds.
    const strong = model.moved.length ? theme.fgMuted : theme.fg;
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";

    const cells = model.report.counts.map((count) => {
      ctx.font = FIGURE_FONT;
      const value = figure(count.value);
      const valueWidth = ctx.measureText(value).width;
      ctx.font = LABEL_FONT;
      return {
        value,
        valueWidth,
        name: count.name,
        width: valueWidth + LABEL_GAP + ctx.measureText(count.name).width,
      };
    });
    if (!cells.length) return;

    const wanted = cells.reduce((total, cell) => total + cell.width, 0)
      + CELL_GAP * (cells.length - 1);
    if (wanted > available) {
      // Too narrow for the numbers at their own size. Every one of them stays on screen, at the
      // smaller size and condensed by the browser, rather than any being dropped.
      ctx.font = LABEL_FONT;
      ctx.fillStyle = strong;
      const joined = cells.map((cell) => `${cell.value} ${cell.name}`).join("   ");
      ctx.fillText(joined, layout.x0, baseline, available);
    } else {
      let x = layout.x0;
      for (const cell of cells) {
        ctx.font = FIGURE_FONT;
        ctx.fillStyle = strong;
        ctx.fillText(cell.value, x, baseline);
        ctx.font = LABEL_FONT;
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(cell.name, x + cell.valueWidth + LABEL_GAP, baseline);
        x += cell.width + CELL_GAP;
      }
    }

    regions.push({
      x: layout.x0,
      y: layout.countsY,
      width: available,
      height: COUNTS_HEIGHT,
      title: COUNTS_HOVER,
    });
  }

  /**
   * What the band of per pattern totals says beside its name.
   *
   * @param {object} model - Model from `readModel`.
   * @param {number} drawn - How many of the totals fitted the band.
   * @returns {{brief: string, full: string}} How many patterns are on screen where some are
   *   not, and otherwise how many of them matched anything.
   */
  function tallyNote(model, drawn) {
    const total = model.talliesTotal;
    if (drawn < total) {
      return {
        brief: `${figure(drawn)} of ${figure(total)}`,
        full: `${figure(drawn)} of ${figure(total)} drawn`,
      };
    }
    // Counted over the whole breakdown, which is all of it on this branch, so the number is
    // never a share of a sample read as a share of the run.
    const matched = model.tallies.filter((entry) => entry.value > 0).length;
    return {
      brief: `${figure(matched)} of ${figure(total)}`,
      full: `${figure(matched)} of ${figure(total)} matched something`,
    };
  }

  /**
   * Draw one per pattern total: the pair on the left, its number on the right.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {{name: string, value: number}} tally - One total from the model.
   * @param {object} box - `{x0, x1}` the band sits in.
   * @param {number} y - Top of the row.
   * @returns {void}
   */
  function drawTally(ctx, theme, tally, box, y) {
    const middle = y + ROW_HEIGHT / 2;
    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    // The number is what the band is read for, so it keeps a column of its own at the right
    // edge and the pair beside it is what gives way on a narrow node.
    const value = figure(tally.value);
    ctx.textAlign = "right";
    ctx.fillStyle = tally.value > 0 ? theme.fg : theme.warning;
    ctx.fillText(value, box.x1 - ROW_PAD, middle);

    const room = box.x1 - ROW_PAD - Math.ceil(ctx.measureText(value).width) - NOTE_GAP
      - (box.x0 + ROW_PAD);
    ctx.textAlign = "left";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(clipText(ctx, tally.name, room), box.x0 + ROW_PAD, middle);
  }

  /**
   * Draw the per pattern totals: the line naming the band, then the rows that fit its box.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {object} box - `{x0, x1, y, height}` the whole band sits in.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawTallies(ctx, theme, model, box, regions) {
    const width = Math.max(1, box.x1 - box.x0);
    const rowsY = box.y + BLOCK_LABEL_HEIGHT;
    const rowsHeight = Math.max(ROW_HEIGHT, box.height - BLOCK_LABEL_HEIGHT);
    const room = Math.max(0, Math.floor((rowsHeight - ROW_PAD) / ROW_HEIGHT));
    const drawn = Math.min(model.tallies.length, room);

    ctx.font = LABEL_FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = theme.fg;
    const nameWidth = ctx.measureText(PATTERNS_LABEL).width;
    ctx.fillText(PATTERNS_LABEL, box.x0, box.y + BLOCK_LABEL_HEIGHT / 2);
    // Shortened, and then given up, rather than drawn over the name. The whole breakdown is on
    // hover however much of it the band has room to draw.
    const note = fitNote(ctx, tallyNote(model, drawn), width - nameWidth - NOTE_GAP);
    if (note) {
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(note, box.x1, box.y + BLOCK_LABEL_HEIGHT / 2);
    }

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(box.x0, rowsY, width, rowsHeight);
    for (let index = 0; index < drawn; index += 1) {
      drawTally(ctx, theme, model.tallies[index], box, rowsY + ROW_PAD + index * ROW_HEIGHT);
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(box.x0 + 0.5, rowsY + 0.5, Math.max(1, width - 1),
      Math.max(1, rowsHeight - 1));

    const held = model.tallies.map((entry) => {
      const matches = entry.value === 1 ? "match" : "matches";
      return `${entry.name}: ${figure(entry.value)} ${matches}`;
    });
    if (drawn < model.tallies.length) {
      held.push(`The band shows ${figure(drawn)} of ${figure(model.tallies.length)} of them. `
        + "Drag the node taller to see more.");
    }
    regions.push({
      x: box.x0,
      y: box.y,
      width,
      height: Math.max(1, box.height),
      title: [...held, PATTERNS_HOVER].join("\n"),
    });
  }

  /**
   * Draw one sample row, with the match marked and the row shifted until the match is on screen.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} row - One row from the model.
   * @param {object} box - The list's geometry.
   * @param {number} y - Top of the row.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawRow(ctx, theme, row, box, y, regions) {
    const middle = y + ROW_HEIGHT / 2;
    ctx.textBaseline = "middle";
    let right = box.x1 - ROW_PAD;
    let note = "";

    if (row.note) {
      ctx.font = LABEL_FONT;
      const noteWidth = Math.ceil(ctx.measureText(row.note).width);
      // The excerpt is the value and the note is what labels it, so the note is the one that
      // gives way when there is no room for both.
      if (right - box.x0 - noteWidth - NOTE_GAP >= MIN_TEXT_WIDTH) {
        note = row.note;
        right -= noteWidth + NOTE_GAP;
      }
    }

    ctx.font = BODY_FONT;
    const span = row.mark ?? [0, 0];
    const before = row.text.slice(0, span[0]);
    const hit = row.text.slice(span[0], span[1]);
    const beforeWidth = ctx.measureText(before).width;
    const hitWidth = ctx.measureText(hit).width;
    const wholeWidth = ctx.measureText(row.text).width;
    const column = right - box.x0;
    // The excerpt carries context on both sides, so on a narrow node the match itself would sit
    // off the right edge. The row slides left until the end of the match is inside the column,
    // and never past the start of the match.
    const shift = Math.max(0, Math.min(beforeWidth, beforeWidth + hitWidth - column + MARK_INSET));
    const left = box.x0 - shift;

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x0, y, Math.max(1, column), ROW_HEIGHT);
    ctx.clip();
    ctx.fillStyle = theme.accent;
    if (hitWidth > 0) {
      ctx.globalAlpha = MARK_ALPHA;
      ctx.fillRect(left + beforeWidth, y + 1, hitWidth, ROW_HEIGHT - 2);
      ctx.globalAlpha = 1;
      ctx.fillRect(left + beforeWidth, y + ROW_HEIGHT - 2 - MARK_RULE, hitWidth, MARK_RULE);
    } else {
      // A match of no width is a position, and a row of context with nothing marked on it
      // would say nothing about which position.
      ctx.fillRect(left + beforeWidth - CARET_WIDTH / 2, y + 1, CARET_WIDTH, ROW_HEIGHT - 2);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(before, left, middle);
    ctx.fillText(row.text.slice(span[1]), left + beforeWidth + hitWidth, middle);
    ctx.fillStyle = theme.fg;
    ctx.fillText(hit, left + beforeWidth, middle);
    ctx.restore();

    // The cut marks are painted over the list's own background after the clip is dropped, so
    // each sits on the row rather than inside the characters it stands for.
    const cutLeft = shift > 0;
    const cutRight = wholeWidth - shift > column;
    const cutWidth = ctx.measureText("...").width + 1;
    ctx.textAlign = "left";
    if (cutLeft) {
      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(box.x0, y, cutWidth, ROW_HEIGHT);
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText("...", box.x0, middle);
    }
    if (cutRight) {
      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(right - cutWidth, y, cutWidth, ROW_HEIGHT);
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText("...", right - cutWidth + 1, middle);
    }

    if (note) {
      ctx.font = LABEL_FONT;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(note, box.x1 - ROW_PAD, middle);
    }

    // Whatever the row could not draw is put where one gesture reaches it: the note the width
    // took away, and the characters either cut mark stands for.
    const held = [];
    if (!note && row.note) held.push(row.note);
    if (cutLeft || cutRight) held.push(row.text);
    if (held.length) {
      regions.push({
        x: box.x0,
        y,
        width: Math.max(1, box.x1 - box.x0),
        height: ROW_HEIGHT,
        title: held.join("\n"),
      });
    }
  }

  /**
   * Draw the sample rows that fit.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {object} box - The list's geometry.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {number} How many rows were drawn.
   */
  function drawRows(ctx, theme, model, box, regions) {
    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(box.x0, box.y, box.x1 - box.x0, box.height);

    const room = Math.max(0, Math.floor((box.height - ROW_PAD) / ROW_HEIGHT));
    const drawn = Math.min(model.rows.length, room);
    for (let index = 0; index < drawn; index += 1) {
      drawRow(ctx, theme, model.rows[index], box, box.y + ROW_PAD + index * ROW_HEIGHT, regions);
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      box.x0 + 0.5,
      box.y + 0.5,
      Math.max(1, box.x1 - box.x0 - 1),
      Math.max(1, box.height - 1),
    );
    regions.push({
      x: box.x0,
      y: box.y,
      width: Math.max(1, box.x1 - box.x0),
      height: Math.max(1, box.height),
      title: model.blocks.length ? ROWS_HOVER : ROWS_ONLY_HOVER,
    });
    return drawn;
  }

  /**
   * Draw one block: the text it holds, wrapped, with every marked span highlighted in place.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} block - One block from the model.
   * @param {object} box - `{x0, x1, y, height}` the whole block sits in.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawBlock(ctx, theme, block, box, regions) {
    const width = Math.max(1, box.x1 - box.x0);
    ctx.textBaseline = "middle";

    ctx.font = LABEL_FONT;
    const middle = box.y + BLOCK_LABEL_HEIGHT / 2;
    ctx.textAlign = "left";
    ctx.fillStyle = theme.fg;
    const nameWidth = ctx.measureText(block.name).width;
    ctx.fillText(block.name, box.x0, middle);
    // The note is shortened, and then given up, rather than squeezed into the name or drawn
    // over it. The whole of it is on hover however much of it is drawn.
    const note = fitNote(ctx, block.note, width - nameWidth - NOTE_GAP);
    if (note) {
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(note, box.x1, middle);
    }

    const textY = box.y + BLOCK_LABEL_HEIGHT;
    const textHeight = Math.max(BLOCK_LINE_HEIGHT, box.height - BLOCK_LABEL_HEIGHT);
    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(box.x0, textY, width, textHeight);

    ctx.font = BODY_FONT;
    // Wrapped rather than scrolled: the panel offers no scrollbar and hands its wheel to the
    // graph, so a line wider than the box would be a line nothing on the node could reach.
    const lines = wrapBody(ctx, block.text, width - BLOCK_PAD * 2);
    const room = Math.max(1, Math.floor((textHeight - BLOCK_PAD) / BLOCK_LINE_HEIGHT));
    const drawn = Math.min(lines.length, room);

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x0, textY, width, textHeight);
    ctx.clip();
    for (let index = 0; index < drawn; index += 1) {
      drawBlockLine(ctx, theme, block, lines, index, {
        left: box.x0 + BLOCK_PAD,
        top: textY + BLOCK_PAD + index * BLOCK_LINE_HEIGHT,
        drawn,
      });
    }
    ctx.restore();

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(box.x0 + 0.5, textY + 0.5, Math.max(1, width - 1),
      Math.max(1, textHeight - 1));

    // An edge the text carries on past is dashed and an edge that is the text's own start or
    // end is solid, so a window onto a large file cannot be read as the file.
    const over = drawn < lines.length;
    if (!block.opens) dashEdge(ctx, theme, box.x0, box.x1, textY + 0.5);
    if (over || !block.ends) dashEdge(ctx, theme, box.x0, box.x1, textY + textHeight - 0.5);

    const held = [block.title];
    if (over) {
      held.push(`The box shows ${figure(drawn)} of ${figure(lines.length)} lines. `
        + "Drag the node taller to see more.");
    }
    if (!block.opens || over || !block.ends) held.push(EDGE_HOVER);
    regions.push({
      x: box.x0,
      y: box.y,
      width,
      height: Math.max(1, box.height),
      title: held.join("\n"),
    });
  }

  /**
   * Draw one line of a block, with the marks that reach it.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context, carrying the block's font.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} block - One block from the model.
   * @param {Array<object>} lines - Every line `wrapBody` answered.
   * @param {number} index - Which of them to draw.
   * @param {object} place - `{left, top, drawn}`: where the line sits and how many are drawn.
   * @returns {void}
   */
  function drawBlockLine(ctx, theme, block, lines, index, place) {
    const line = lines[index];
    const stop = line.start + line.text.length;
    const next = index + 1 < place.drawn ? lines[index + 1].start : -1;
    const { spans, carets } = lineMarks(block.marks, line.start, stop, next);
    const at = markOffsets(ctx, line.text, spans, carets);
    const baseline = place.top + BLOCK_LINE_HEIGHT / 2;

    ctx.fillStyle = theme.accent;
    for (const [first, last] of spans) {
      const x = place.left + at.get(first);
      const span = Math.max(1, at.get(last) - at.get(first));
      ctx.globalAlpha = MARK_ALPHA;
      ctx.fillRect(x, place.top, span, BLOCK_LINE_HEIGHT - 1);
      ctx.globalAlpha = 1;
      ctx.fillRect(x, place.top + BLOCK_LINE_HEIGHT - 1 - MARK_RULE, span, MARK_RULE);
    }
    for (const caret of carets) {
      ctx.fillRect(place.left + at.get(caret) - CARET_WIDTH / 2, place.top, CARET_WIDTH,
        BLOCK_LINE_HEIGHT - 1);
    }

    ctx.textAlign = "left";
    let cursor = 0;
    for (const [first, last] of spans) {
      if (first > cursor) {
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(line.text.slice(cursor, first), place.left + at.get(cursor), baseline);
      }
      ctx.fillStyle = theme.fg;
      ctx.fillText(line.text.slice(first, last), place.left + at.get(first), baseline);
      cursor = last;
    }
    if (cursor < line.text.length) {
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(line.text.slice(cursor), place.left + at.get(cursor), baseline);
    }
  }

  /**
   * Draw one line of words in the middle of a band.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {string} text - The words.
   * @param {string} colour - What to draw them in.
   * @param {object} band - `{x0, x1, y, height}` in element pixels.
   * @returns {void}
   */
  function drawNotice(ctx, text, colour, band) {
    ctx.font = BODY_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = colour;
    ctx.fillText(
      text,
      (band.x0 + band.x1) / 2,
      band.y + band.height / 2,
      Math.max(8, band.x1 - band.x0 - 8),
    );
  }

  /**
   * Draw the footer, and collect the regions its hover text sits in.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {number} drawn - How many sample rows were drawn.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model, drawn, regions) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    ctx.font = LABEL_FONT;
    ctx.textBaseline = "middle";

    const box = drawIcon(
      ctx,
      model.claim.icon,
      layout.x0,
      layout.footerY + (layout.footerHeight - ICON_SIZE) / 2,
      ICON_SIZE,
      model.claim.icon === ICON.WARNING ? theme.warning : theme.fgMuted,
    );
    regions.push({ ...box, title: iconTitle(model.claim.icon, model.claim.detail) });
    const glyphWidth = ICON_SIZE + GLYPH_GAP;

    let rightWidth = 0;
    if (drawn) {
      // How many of the matches the rows stand for. With blocks above them the rows are the
      // matches those blocks do not reach, which is a different number from every match found.
      const beyond = model.report.itemsTotal;
      let text = "";
      if (model.blocks.length) {
        text = drawn < beyond
          ? `${figure(drawn)} of ${figure(beyond)} past the blocks`
          : `${figure(beyond)} past the blocks`;
      } else if (drawn < beyond) {
        text = `first ${figure(drawn)} ${drawn === 1 ? "match" : "matches"}`;
      }
      if (text) {
        rightWidth = ctx.measureText(text).width + NOTE_GAP;
        ctx.textAlign = "right";
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(text, layout.x1, middle);
      }
    }

    const facts = model.report.facts.map((entry) => `${entry.name}: ${entry.value}`).join("   ");
    const available = layout.x1 - layout.x0 - glyphWidth - rightWidth;
    if (facts && available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(facts, layout.x0 + glyphWidth, middle, available);
    }

    regions.push({
      x: layout.x0,
      y: layout.footerY,
      width: layout.x1 - layout.x0,
      height: layout.footerHeight,
      title: FOOTER_HOVER,
    });
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return;

    // The graph's zoom is in here as well as the screen's density, so a magnified node is drawn
    // at the resolution it is shown at. Everything below `setTransform` stays in layout units.
    const ratio = surfaceRatio(root);
    const deviceWidth = Math.max(1, Math.round(width * ratio));
    const deviceHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== deviceWidth) canvas.width = deviceWidth;
    if (canvas.height !== deviceHeight) canvas.height = deviceHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    state.layout = computeLayout(width, height);
    const layout = state.layout;
    const theme = readTheme();
    const model = readModel(theme);
    const regions = [];

    if (!model.report) {
      const band = {
        x0: layout.x0,
        x1: layout.x1,
        y: layout.countsY,
        height: layout.footerY - layout.countsY,
      };
      if (model.notice) drawNotice(ctx, model.notice, theme.fgMuted, band);
      if (model.noticeTitle) {
        regions.push({
          x: band.x0,
          y: band.y,
          width: band.x1 - band.x0,
          height: band.height,
          title: model.noticeTitle,
        });
      }
      titles.set(regions);
      return;
    }

    drawCounts(ctx, theme, model, regions);

    // The line takes its own band above the body where both fit, and the whole body where there
    // is nothing else to show. It is what somebody has to act on, so the blocks and the rows
    // give way to it rather than the other way round.
    const listed = model.blocks.length || model.rows.length || model.tallies.length;
    const stacked = model.line && listed
      && layout.bodyHeight >= STATE_HEIGHT + BLOCK_LABEL_HEIGHT + BLOCK_LINE_HEIGHT;
    let bodyY = layout.bodyY + (stacked ? STATE_HEIGHT : 0);
    let bodyHeight = layout.bodyHeight - (stacked ? STATE_HEIGHT : 0);
    let drawn = 0;

    if (model.line) {
      const band = stacked
        ? { x0: layout.x0, x1: layout.x1, y: layout.bodyY, height: STATE_HEIGHT }
        : { x0: layout.x0, x1: layout.x1, y: layout.bodyY, height: layout.bodyHeight };
      if (stacked) {
        ctx.font = BODY_FONT;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = model.tone;
        ctx.fillText(model.line, band.x0, band.y + band.height / 2, band.x1 - band.x0);
      } else {
        drawNotice(ctx, model.line, model.tone, band);
      }
      if (model.lineTitle) {
        regions.push({
          x: band.x0,
          y: band.y,
          width: band.x1 - band.x0,
          height: band.height,
          title: model.lineTitle,
        });
      }
    }

    if (listed && (stacked || !model.line)) {
      const least = model.blocks.length
        * (BLOCK_LABEL_HEIGHT + BLOCK_LINE_HEIGHT + BAND_GAP);

      // The per pattern totals open the body: they are the answer to which pattern fired, they
      // are a few short rows, and the blocks below them keep a floor of their own.
      if (model.tallies.length) {
        const asked = BLOCK_LABEL_HEIGHT + ROW_PAD * 2 + model.tallies.length * ROW_HEIGHT;
        const bandHeight = Math.min(
          asked,
          Math.floor(bodyHeight * PATTERNS_SHARE),
          bodyHeight - least,
        );
        if (bandHeight >= BLOCK_LABEL_HEIGHT + ROW_PAD + ROW_HEIGHT) {
          drawTallies(
            ctx,
            theme,
            model,
            { x0: layout.x0, x1: layout.x1, y: bodyY, height: bandHeight },
            regions,
          );
          bodyY += bandHeight + BAND_GAP;
          bodyHeight -= bandHeight + BAND_GAP;
        }
      }

      // The rows are a sample of the matches the blocks do not reach, so they take a share of
      // the room and only what they can fill of it. What the blocks need for a name and one
      // line each comes off that share before it is taken, since a panel of rows with no
      // blocks above them is the repeated excerpts the blocks were built to replace.
      const wanted = model.rows.length ? model.rows.length * ROW_HEIGHT + ROW_PAD * 2 : 0;
      const listHeight = model.blocks.length
        ? Math.min(wanted, Math.floor(bodyHeight * ROWS_SHARE), bodyHeight - least)
        : bodyHeight;
      if (model.rows.length && listHeight >= ROW_HEIGHT + ROW_PAD) {
        drawn = drawRows(
          ctx,
          theme,
          model,
          {
            x0: layout.x0,
            x1: layout.x1,
            y: bodyY + bodyHeight - listHeight,
            height: listHeight,
          },
          regions,
        );
        bodyHeight -= listHeight + BAND_GAP;
      }
      const room = model.blocks.length;
      const each = room ? Math.floor((bodyHeight - BAND_GAP * (room - 1)) / room) : 0;
      if (each >= BLOCK_LABEL_HEIGHT + BLOCK_LINE_HEIGHT) {
        for (let index = 0; index < room; index += 1) {
          drawBlock(
            ctx,
            theme,
            model.blocks[index],
            {
              x0: layout.x0,
              x1: layout.x1,
              y: bodyY + index * (each + BAND_GAP),
              height: each,
            },
            regions,
          );
        }
      }
    }

    drawFooter(ctx, theme, model, drawn, regions);
    titles.set(regions);
  }

  /**
   * Repaint on the next frame, coalescing repeated requests into one.
   *
   * @returns {void}
   */
  function schedulePaint() {
    if (state.disposed || state.paintHandle) return;
    state.paintHandle = requestAnimationFrame(() => {
      state.paintHandle = 0;
      try {
        paint();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the run readout:`, error);
      }
    });
  }

  /**
   * Wrap an event handler so a failure is logged rather than thrown at the browser.
   *
   * @param {(event: Event) => void} handler - Handler to wrap.
   * @returns {(event: Event) => void} The wrapped handler.
   */
  function guard(handler) {
    return (event) => {
      try {
        handler(event);
      } catch (error) {
        console.error(`[${EXT_NAME}] Run readout input failed:`, error);
      }
    };
  }

  /**
   * Ask again once a saved workflow has been applied.
   *
   * @returns {void}
   */
  function handleConfigured() {
    if (state.mounted) refresh();
    schedulePaint();
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) app.canvas?.processMouseDown?.(event);
  };

  const onPointerUp = (event) => {
    if (event.button === 1) app.canvas?.processMouseUp?.(event);
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  };

  const onPointerEnter = () => {
    // A run whose end event was missed, and a socket that opened after the last answer, both
    // show up as an answer that has been standing for a while.
    if (state.mounted && Date.now() - state.fetchedAt > STALE_MS) refresh();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointerenter", guard(onPointerEnter));
  root.addEventListener("contextmenu", guard(onContextMenu));
  // The panel scrolls nothing, so it takes every wheel gesture over it and the graph zooms
  // from the canvas around the node.
  const releaseWheel = captureWheel(root);

  /**
   * Take the listeners and ask for the report, once the element is really on a node.
   *
   * @returns {void}
   */
  function mount() {
    if (state.mounted || state.disposed) return;
    state.mounted = true;
    joinRuns();
    refresh();
  }

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => {
      // A box is the one proof the element is in the document. A node copied to the clipboard
      // builds a panel that never gets one and is never removed.
      if (root.clientWidth > 0 && root.clientHeight > 0) mount();
      schedulePaint();
    });
    observer.observe(root);
  } else {
    // With no observer there is no signal to wait for. Every browser the frontend supports has
    // one, so this is the path a panel takes rather than never being drawn.
    mount();
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here.
  let unwatchRatio = watchSurfaceRatio(root, schedulePaint);

  // The panel is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the listeners, observers and hover text the panel holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    state.paintHandle = 0;
    // The token moves past every answer still in flight, so none of them is taken after this.
    state.token += 1;
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
    state.unwatchRuns?.();
    state.unwatchRuns = null;
    titles.dispose();
  }

  // The setting names the height the panel starts at; the maximum is open, so the node's spare room
  // reaches the readout. `boundTextBoxes` gives every multiline box a ceiling:
  // the frontend divides spare room between every widget whose maximum is above its minimum, and a
  // multiline string widget declares no maximum at all, so unbounded the seventeen of them would
  // take the whole of every drag and the readout none of it.
  return {
    element: root,
    height,
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    handleConfigured,
    dispose,
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
 * Append the readout to a node and wire it to the widgets it reports on.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachRunReadout(node) {
  for (const name of INPUT_NAMES) {
    if (!findWidget(node, name)) return;
  }

  const readout = createRunReadout(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, readout, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // After the panel is appended, so every widget the readout competes with for the node's spare
  // room is present to be bounded.
  boundTextBoxes(node);

  for (const name of INPUT_NAMES) {
    chainWidgetCallback(node, name, readout.schedulePaint);
  }

  // A link is attached and detached with no widget callback of its own, and either changes what
  // the next run reads.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      readout.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a link changed:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      readout.handleConfigured();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the report after a workflow load:`, error);
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
      readout.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the run readout:`, error);
    }
    return result;
  };

  readout.schedulePaint();
}

// The find and replace boxes, one group per pair, in the order the schema declares them. The
// first pair is named without a number, the rest carry one.
const PAIR_NAME = /^(find|replace)_(\d+)$/;

/**
 * The node's find and replace boxes, grouped into pairs.
 *
 * @param {object} node - The node to read.
 * @returns {string[][]} One `[find, replace]` per pair, first pair first.
 */
function pairGroups(node) {
  const numbered = new Map();
  for (const widget of node.widgets ?? []) {
    const parsed = PAIR_NAME.exec(widget.name ?? "");
    if (!parsed) continue;
    const index = Number(parsed[2]);
    numbered.set(index, [...(numbered.get(index) ?? []), widget.name]);
  }
  const order = [...numbered.keys()].sort((a, b) => a - b);
  return [["find", "replace"], ...order.map((index) => numbered.get(index))];
}


app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Text Find and Replace", "Run readout"],
      name: "Show the run readout",
      tooltip:
        "Draw the counts, one total per pattern, the text as it was searched and as it came out "
        + "with every match marked, and the input the run read, under the widgets of Text Find "
        + "and Replace. The node's outputs are the same either way. This applies to nodes added "
        + "after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
    {
      id: HEIGHT_SETTING_ID,
      category: ["WAS Node Suite", "Text Find and Replace", "Readout height"],
      name: "Run readout height in pixels",
      tooltip:
        "Pixels of node given to the readout, from 148 to 900. Taller shows more patterns and "
        + "more of the two blocks of text at once. This applies to nodes added after the setting "
        + "changes, so a reload applies it to the nodes already on the canvas.",
      type: "slider",
      attrs: { min: MIN_HEIGHT, max: MAX_HEIGHT, step: 4 },
      defaultValue: DEFAULT_HEIGHT,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second panel.
    if (proto.__was_search_and_replace_wrapped) return;
    proto.__was_search_and_replace_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachRunReadout(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the run readout:`, error);
      }
      try {
        growWidgets(this, pairGroups(this));
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to fold the find and replace pairs:`, error);
      }
      return result;
    };
  },
});
