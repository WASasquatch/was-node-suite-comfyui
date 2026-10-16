/**
 * Nova filter preview for the Image Nova Filter node.
 *
 * Draws the node's 256 entry lookup table per channel twice: a picture before and after it, and
 * the curve itself over a strip of the grey ramp it produces.
 */

// This file sits at the top of `web/`, so ComfyUI's own modules are reached with
// `../../scripts/`, one level shallower than the shared components under `web/interface/`. A
// specifier one level too deep resolves under `/extensions/`, where nothing answers, and the
// frontend logs the failed import and carries on without the extension.
import { app } from "../../scripts/app.js";
import { imageBackdrop } from "./interface/backdrop.js";
import {
  BACKDROP_KIND,
  MAGNIFIED_AT,
  REDUCED_AT,
  TEST_CARD,
  appendFilterWidget,
  createFilterSurface,
} from "./interface/filter_surface.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { PREVIEW_STATE } from "./interface/preview.js";
import { floorMod, truncate } from "./interface/python_arithmetic.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeDone } from "./interface/run_events.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.NovaFilterUI";
const NODE_NAME = "Image Nova Filter";
const SETTING_ID = "WAS.NovaFilter.ShowInterface";

const AMPLITUDE = "amplitude";
const FREQUENCY = "frequency";
const WIDGETS = [AMPLITUDE, FREQUENCY];

// The schema's own defaults, read only when a widget cannot be.
const DEFAULTS = {
  [AMPLITUDE]: 0.1,
  [FREQUENCY]: 3.14,
};

const CURVE_WIDGET_NAME = "was_nova_curve_ui";
const CURVE_WIDGET_TYPE = "was_nova_curve";

// Height of the two appended widgets in node units. The surface is tall enough for the whole
// test card at one device pixel per card pixel: its panels are the element height less the
// header, the footer and the padding, and the card is 192 tall.
const SURFACE_HEIGHT = 260;
const CURVE_HEIGHT = 136;

// A DOM widget element is inset by the widget's margin on every side, so the element itself is
// shorter than the widget by twice that margin.
const UI_MARGIN = 10;
const CURVE_ELEMENT_HEIGHT = CURVE_HEIGHT - UI_MARGIN * 2;

// The node reads an 8 bit image, so 256 entries is the whole of what the filter can be asked
// and the whole of what it can answer, not a sample of it.
const LEVELS = 256;
const MAX_LEVEL = 255;

// Layout bands of the curve, measured in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const GUTTER_WIDTH = 18;
const STRIP_HEIGHT = 10;
const STRIP_GAP = 2;
const TICK_HEIGHT = 10;
const FOOTER_HEIGHT = 13;

// The footer's one line. It opens with the glyph for what the plot leaves out, then carries the
// readout, and on the right a linked input or a capped frequency.
const FOOTER_LINES = 1;
const MIN_PLOT_HEIGHT = 24;

// The gap kept between the glyph and the readout beside it.
const GLYPH_GAP = 4;

const BODY_FONT = "10px sans-serif";
const AXIS_FONT = "9px sans-serif";

const SAMPLE_THICKNESS = 1.6;
const GRID_ALPHA = 0.4;

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
 * Read one widget as a number.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {number} The value the widget holds, or the schema's default for it.
 */
function widgetNumber(node, name) {
  const value = Number(findWidget(node, name)?.value);
  return Number.isFinite(value) ? value : DEFAULTS[name];
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
 * Format a number for a footer.
 *
 * @param {number} value - Value to write.
 * @returns {string} The value with at most three decimals, which is the step both widgets take.
 */
function formatNumber(value) {
  if (!Number.isFinite(value)) return "?";
  return String(Math.round(value * 1000) / 1000);
}

/**
 * Build the lookup table the node applies, and the three numbers that describe it.
 *
 * Each entry is `nova_sine` for one input level.
 *
 * @param {number} amplitude - Peak height of the wave, where 1 is the whole black to white
 *   range.
 * @param {number} frequency - Cycles of the wave across the brightness range, already capped.
 * @returns {{lut: Uint8Array, wraps: number, peak: number}} The table, the number of places two
 *   neighbouring levels sit on opposite sides of black, and the furthest the wave travels from
 *   black in either direction.
 */
function buildTable(amplitude, frequency) {
  const lut = new Uint8Array(LEVELS);
  let wraps = 0;
  let peak = 0;
  let below = false;

  for (let level = 0; level < LEVELS; level++) {
    // `amplitude * sin(2 * pi * frequency * (level / 255)) * 255`, composed in the node's own
    // order, so the two sides round the same intermediate products.
    const scaled = amplitude * Math.sin(2 * Math.PI * frequency * (level / MAX_LEVEL)) * 255;
    // Truncated toward zero, which is what `astype(np.int64)` does. `Math.round` disagrees at the
    // crest of a wave, where one cycle over 256 levels reaches 254.95 and the node keeps 254, and
    // `Math.floor` disagrees over the whole negative half.
    const whole = truncate(scaled);
    // The wrap to white is the whole of this line: a level below black comes back around the top
    // of the range, which is what storing a negative value into a `uint8` array does. Reducing
    // modulo 256 takes the sign of the divisor, which is what numpy and Python do and what
    // JavaScript's `%` does not: `-1 % 256` is 255 in Python and -1 in JavaScript, so a plain
    // remainder would draw the effect away.
    lut[level] = floorMod(whole, LEVELS);

    const negative = whole < 0;
    if (level > 0 && negative !== below) wraps += 1;
    below = negative;
    peak = Math.max(peak, Math.abs(whole));
  }

  return { lut, wraps, peak };
}

/**
 * Read everything the two drawings need from the node.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {number} sourceWidth - Width of the image the filter is applied to, in its own pixels.
 * @returns {{amplitude: number, asked: number, frequency: number, capped: boolean,
 *   lut: Uint8Array, wraps: number, peak: number, linked: string[]}} The two values as held, the
 *   frequency after the node's own cap, the table it produces, and which inputs a link fills in.
 */
function readModel(node, sourceWidth) {
  const amplitude = widgetNumber(node, AMPLITUDE);
  const asked = widgetNumber(node, FREQUENCY);
  // The node caps the frequency at half the width of the image it is given, past which the
  // bands are finer than the pixels that would carry them.
  const cap = Math.max(0, sourceWidth) / 2;
  const frequency = asked > cap ? cap : asked;

  return {
    amplitude,
    asked,
    frequency,
    capped: frequency !== asked,
    ...buildTable(amplitude, frequency),
    linked: WIDGETS.filter((name) => inputLinked(node, name)),
  };
}

/**
 * What a linked input has to say, in the words both footers use.
 *
 * @param {string[]} linked - Names of the inputs a link fills in.
 * @returns {string} The note, empty when every value on screen is one the run reads.
 */
function linkedNote(linked) {
  if (linked.length === 1) return `${linked[0]} is linked`;
  if (linked.length > 1) return `${linked.length} inputs are linked`;
  return "";
}

/**
 * Build the curve for one node.
 *
 * @param {object} node - The node the curve is drawn on.
 * @param {() => object} read - Answers the model from `readModel` for the current settings.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   dispose: () => void}} The element to hand to `addDOMWidget`, the height it was built for, a
 *   coalesced repaint, and teardown.
 */
function createNovaCurve(node, read) {
  const root = document.createElement("div");
  // No `tabIndex`. The curve reads out and writes nothing, so it needs no focus, and an element
  // that takes focus inside the graph canvas has to consume every key ComfyUI binds, Delete and
  // Backspace included, or a keystroke deletes the node it is drawn on.
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${CURVE_ELEMENT_HEIGHT}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The footer's glyph states its sentence through the element's own title. The region is handed
  // over again on every repaint, since the glyph moves whenever the node is resized.
  const titles = hoverTitles(root);

  const state = {
    hoverLevel: null,
    strip: null,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Work out where the plot, the strip, the ticks and the footer sit inside the element.
   *
   * @param {number} width - Element width in pixels.
   * @param {number} height - Element height in pixels.
   * @returns {object} Pixel geometry of each band.
   */
  function computeLayout(width, height) {
    const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT * FOOTER_LINES);
    const tickY = Math.max(0, footerY - TICK_HEIGHT);
    const stripY = Math.max(0, tickY - STRIP_HEIGHT);
    const plotX0 = PAD_X + GUTTER_WIDTH;
    const plotX1 = Math.max(plotX0 + 1, width - PAD_X);
    const plotY0 = PAD_Y;
    const plotY1 = Math.max(plotY0 + MIN_PLOT_HEIGHT, stripY - STRIP_GAP);

    return {
      width,
      height,
      plotX0,
      plotX1,
      plotY0,
      plotY1,
      plotWidth: plotX1 - plotX0,
      plotHeight: plotY1 - plotY0,
      cellWidth: (plotX1 - plotX0) / LEVELS,
      cellHeight: (plotY1 - plotY0) / LEVELS,
      stripY,
      tickY,
      footerY,
    };
  }

  /**
   * Where the left edge of one level's column sits.
   *
   * @param {number} level - Input level, 0 to 256.
   * @returns {number} Position in element pixels.
   */
  function xFromEdge(level) {
    return state.layout.plotX0 + level * state.layout.cellWidth;
  }

  /**
   * Where the middle of one level's column sits.
   *
   * @param {number} level - Input level.
   * @returns {number} Position in element pixels.
   */
  function xFromLevel(level) {
    return xFromEdge(level) + state.layout.cellWidth / 2;
  }

  /**
   * Where one output level sits, measured up from the bottom of the plot.
   *
   * @param {number} output - Output level.
   * @returns {number} Position in element pixels.
   */
  function yFromOutput(output) {
    return state.layout.plotY1 - (output + 0.5) * state.layout.cellHeight;
  }

  /**
   * Which input level a position across the plot stands for.
   *
   * @param {number} x - Position in element pixels.
   * @returns {number|null} The level, or null when the position is outside the plot.
   */
  function levelFromX(x) {
    const layout = state.layout;
    if (x < layout.plotX0 || x > layout.plotX1 || !(layout.cellWidth > 0)) return null;
    return clamp(Math.floor((x - layout.plotX0) / layout.cellWidth), 0, MAX_LEVEL);
  }

  /**
   * Read the pointer position in element pixels.
   *
   * @param {PointerEvent|MouseEvent} event - Event to read.
   * @returns {{x: number, y: number}} Position inside the element.
   */
  function localPoint(event) {
    return elementPoint(root, event);
  }

  /**
   * Draw the quarter grid and the line an untouched image would follow.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawGrid(ctx, theme) {
    const layout = state.layout;

    ctx.globalAlpha = GRID_ALPHA;
    ctx.strokeStyle = theme.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const level of [64, 128, 192]) {
      const x = Math.round(xFromEdge(level)) + 0.5;
      ctx.moveTo(x, layout.plotY0);
      ctx.lineTo(x, layout.plotY1);
      const y = Math.round(yFromOutput(level)) + 0.5;
      ctx.moveTo(layout.plotX0, y);
      ctx.lineTo(layout.plotX1, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = theme.fgMuted;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(xFromLevel(0), yFromOutput(0));
    ctx.lineTo(xFromLevel(MAX_LEVEL), yFromOutput(MAX_LEVEL));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw one cell per input level. Cells are never joined.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {Uint8Array} lut - The lookup table from `buildTable`.
   * @returns {void}
   */
  function drawCurve(ctx, theme, lut) {
    const layout = state.layout;
    const width = Math.max(layout.cellWidth, 1);
    const height = Math.max(layout.cellHeight, SAMPLE_THICKNESS);

    ctx.beginPath();
    // A rectangle per level rather than a path through them. The wave carries a run of levels
    // below black and every one of them comes back near white, so the curve genuinely jumps from
    // one end of the range to the other between two neighbouring inputs, and a line drawn through
    // that jump would claim output levels the node never produces.
    for (let level = 0; level < LEVELS; level++) {
      ctx.rect(xFromLevel(level) - width / 2, yFromOutput(lut[level]) - height / 2, width, height);
    }
    ctx.fillStyle = theme.fg;
    ctx.fill();
  }

  /**
   * Draw the grey ramp the table turns a grey ramp into, one device pixel column at a time.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {number} ratio - Device pixels per element pixel.
   * @param {Uint8Array} lut - The lookup table from `buildTable`.
   * @returns {void}
   */
  function drawStrip(ctx, ratio, lut) {
    const layout = state.layout;
    const width = Math.max(1, Math.round(layout.plotWidth * ratio));
    const height = Math.max(1, Math.round(STRIP_HEIGHT * ratio));

    // Both sides carry the zoom, so the buffer costs the square of it: at the default node
    // width, magnified until the ratio's own budget stops it, that is 1993 by 78, which is 600
    // kilobytes a frame where a pointer crossing the plot repaints for every level it passes
    // over. It is held between repaints instead, since every pixel in it is written below
    // before any of it is drawn.
    let image = state.strip;
    if (!image || image.width !== width || image.height !== height) {
      image = ctx.createImageData(width, height);
      state.strip = image;
    }
    const row = new Uint8ClampedArray(width * 4);

    // Each column takes its level straight from the table, so no two levels are blended into a
    // colour the node cannot produce. The bands and the hard edge at each wrap are what this
    // strip is for, and they are the part of the effect a curve alone reads as a gap.
    for (let column = 0; column < width; column++) {
      const level = clamp(Math.floor((column * LEVELS) / width), 0, MAX_LEVEL);
      const value = lut[level];
      const offset = column * 4;
      row[offset] = value;
      row[offset + 1] = value;
      row[offset + 2] = value;
      row[offset + 3] = 255;
    }

    for (let line = 0; line < height; line++) image.data.set(row, line * width * 4);
    ctx.putImageData(image, Math.round(layout.plotX0 * ratio), Math.round(layout.stripY * ratio));
  }

  /**
   * Mark the level the pointer is over, on the plot and on the strip.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {Uint8Array} lut - The lookup table from `buildTable`.
   * @returns {void}
   */
  function drawHover(ctx, theme, lut) {
    const level = state.hoverLevel;
    if (level === null) return;
    const layout = state.layout;
    const x = Math.round(xFromLevel(level)) + 0.5;

    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, layout.plotY0);
    ctx.lineTo(x, layout.plotY1);
    ctx.stroke();
    ctx.strokeRect(
      Math.round(xFromEdge(level)) + 0.5,
      Math.round(layout.stripY) + 0.5,
      Math.max(1, Math.round(layout.cellWidth)),
      Math.max(1, STRIP_HEIGHT - 1),
    );

    ctx.fillStyle = theme.accent;
    const height = Math.max(layout.cellHeight, SAMPLE_THICKNESS);
    ctx.fillRect(
      xFromLevel(level) - Math.max(layout.cellWidth, 2) / 2,
      yFromOutput(lut[level]) - Math.max(height, 2) / 2,
      Math.max(layout.cellWidth, 2),
      Math.max(height, 2),
    );
  }

  /**
   * Draw the two axes.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawAxes(ctx, theme) {
    const layout = state.layout;

    ctx.font = AXIS_FONT;
    ctx.fillStyle = theme.fgMuted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const level of [MAX_LEVEL, 128, 0]) {
      // The top and bottom rows of the lattice are half a cell from the edge of the plot, so
      // their labels are held inside it rather than drawn half outside.
      const y = clamp(yFromOutput(level), layout.plotY0 + 4, layout.plotY1 - 4);
      ctx.fillText(String(level), layout.plotX0 - 2, y);
    }

    const middle = layout.tickY + TICK_HEIGHT / 2;
    ctx.textAlign = "left";
    ctx.fillText("0", layout.plotX0, middle);
    ctx.textAlign = "center";
    ctx.fillText("128", xFromEdge(128), middle);
    ctx.textAlign = "right";
    ctx.fillText("255", layout.plotX1, middle);
  }

  /**
   * The left half of the footer line: the level under the pointer, or the wrap count and the
   * peak.
   *
   * @param {object} model - Model from `readModel`.
   * @returns {string} Text to draw.
   */
  function footerReadout(model) {
    if (state.hoverLevel !== null) {
      return `in ${state.hoverLevel}   out ${model.lut[state.hoverLevel]}`;
    }
    // Both a zero amplitude and a zero frequency render a black image, so a wave that never
    // leaves black says so rather than reading as a curve with nothing worth noting on it.
    if (model.peak === 0) return "every level to 0, the image renders black";
    // The two numbers the widgets cannot give: how many times the wave crosses black, which is
    // how many hard edges the image gains, and how far from black it ever travels, which is how
    // dark the result stays.
    const wraps = `${model.wraps} ${model.wraps === 1 ? "wrap" : "wraps"}`;
    return `${wraps}   peak ${model.peak} of 255`;
  }

  /**
   * The note the footer line carries on the right: a linked input, then a capped frequency.
   *
   * @param {object} model - Model from `readModel`.
   * @returns {string} The note, empty when there is nothing to report.
   */
  function footerNote(model) {
    // A linked input leads, since a number the run does not read makes every other reading on
    // screen provisional.
    const linked = linkedNote(model.linked);
    // The cap follows it rather than waiting for it to clear: the curve is plotted from the
    // capped frequency whether an input is linked or not, and this is the only place that number
    // is named.
    const capped = model.capped
      ? `frequency capped to ${formatNumber(model.frequency)} by the image`
      : "";
    if (linked && capped) return `${linked}, ${capped}`;
    return linked || capped;
  }

  /**
   * Draw the footer line.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readModel`.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model) {
    const layout = state.layout;
    const middle = layout.footerY + FOOTER_HEIGHT / 2;
    const note = footerNote(model);

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    // The curve is drawn in grey and the node applies it to red, green and blue on their own, so
    // a colour comes apart into three different points on this one curve. That is the whole of
    // what the plot leaves out and it is in force whatever the settings are, so it opens the line
    // as a glyph rather than as words that would take the readout's room every frame.
    const box = drawIcon(
      ctx,
      ICON.APPROXIMATE,
      layout.plotX0,
      middle - ICON_SIZE / 2,
      ICON_SIZE,
      theme.fgMuted,
    );
    titles.set([
      {
        ...box,
        title: iconTitle(ICON.APPROXIMATE, "Per channel"),
      },
    ]);
    const glyphWidth = ICON_SIZE + GLYPH_GAP;

    let noteWidth = 0;
    if (note) {
      noteWidth = ctx.measureText(note).width;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.warning;
      ctx.fillText(note, layout.plotX1, middle);
    }

    const available = layout.plotWidth - glyphWidth - noteWidth - 8;
    if (available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = state.hoverLevel === null ? theme.fgMuted : theme.fg;
      ctx.fillText(footerReadout(model), layout.plotX0 + glyphWidth, middle, available);
    }
  }

  /**
   * Draw the whole curve.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return;

    // The graph's zoom is in here as well as the screen's density, so a magnified node is drawn
    // at the resolution it is shown at. Everything below `setTransform` stays in layout units
    // bar the strip, which writes device pixels straight in and is handed this same number.
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
    const model = read();

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.plotX0, layout.plotY0, layout.plotWidth, layout.plotHeight);

    drawGrid(ctx, theme);
    drawCurve(ctx, theme, model.lut);
    drawStrip(ctx, ratio, model.lut);
    drawHover(ctx, theme, model.lut);

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      Math.round(layout.plotX0) + 0.5,
      Math.round(layout.plotY0) + 0.5,
      Math.max(1, Math.round(layout.plotWidth) - 1),
      Math.max(1, Math.round(layout.plotHeight) - 1),
    );
    ctx.strokeRect(
      Math.round(layout.plotX0) + 0.5,
      Math.round(layout.stripY) + 0.5,
      Math.max(1, Math.round(layout.plotWidth) - 1),
      Math.max(1, STRIP_HEIGHT - 1),
    );

    drawAxes(ctx, theme);
    drawFooter(ctx, theme, model);
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
        console.error(`[${EXT_NAME}] Failed to draw the nova curve:`, error);
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
        console.error(`[${EXT_NAME}] Nova curve input failed:`, error);
      }
    };
  }

  /**
   * Follow the pointer across the plot, reading out the level it is over.
   *
   * @param {PointerEvent} event - The move to read.
   * @returns {void}
   */
  function onPointerMove(event) {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }
    const point = localPoint(event);
    const layout = state.layout;
    const inside = point.y >= layout.plotY0 && point.y <= layout.stripY + STRIP_HEIGHT;
    const level = inside ? levelFromX(point.x) : null;
    if (level === state.hoverLevel) return;
    state.hoverLevel = level;
    schedulePaint();
  }

  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener(
    "pointerleave",
    guard(() => {
      if (state.hoverLevel === null) return;
      state.hoverLevel = null;
      schedulePaint();
    }),
  );
  root.addEventListener(
    "pointerdown",
    guard((event) => {
      // Middle button panning belongs to the canvas underneath. Nothing else here is a gesture.
      if (event.button === 1) app.canvas?.processMouseDown?.(event);
    }),
  );
  root.addEventListener(
    "pointerup",
    guard((event) => {
      if (event.button === 1) app.canvas?.processMouseUp?.(event);
    }),
  );
  root.addEventListener(
    "contextmenu",
    guard((event) => {
      // The graph canvas suppresses its own context menu on its own element, and this is a
      // separate element, so the browser menu would otherwise open over the node.
      event.preventDefault();
      event.stopPropagation();
    }),
  );
  // The curve scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here. The two answer different events: the observer answers a
  // node that was resized or collapsed, this answers the same box drawn at another size.
  let unwatchRatio = watchSurfaceRatio(root, schedulePaint);

  // The panel is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the frame the curve has asked for, the buffer it holds, what it watches and its
   * hover text.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    state.paintHandle = 0;
    state.strip = null;
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
    titles.dispose();
  }

  return {
    element: root,
    height: CURVE_HEIGHT,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    dispose,
  };
}

/**
 * The published picture, with the width of the image it was reduced from kept beside it.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {{width: number}} source - Filled in with the image's width on every answer, and
 *   emptied whenever there is no picture.
 * @param {() => void} onChange - Called whenever that width becomes a different number.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function trackedBackdrop(node, source, onChange) {
  // The curve is drawn from this same width and is repainted by nothing the surface does, so a
  // width that arrives after the curve was last drawn would leave the two panels of one interface
  // contradicting each other about the cap until something unrelated redrew it. The change is
  // reported rather than the width being written silently.
  const write = (width) => {
    if (source.width === width) return;
    source.width = width;
    onChange?.();
  };

  // The width is the one thing outside the two widgets that changes what the node computes, since
  // the frequency is capped at half of it. It is taken from the answer the surface is given
  // rather than asked for a second time, so the picture and the cap can never be reading
  // different images.
  const backdrop = imageBackdrop(node);
  return {
    async load() {
      try {
        const answer = await backdrop.load();
        const ready = answer?.state === PREVIEW_STATE.READY && Number(answer?.width) > 0;
        write(ready ? Number(answer.width) : 0);
        return answer;
      } catch (error) {
        write(0);
        throw error;
      }
    },
  };
}

/**
 * What this preview does not reproduce, measured against the picture on screen.
 *
 * @param {object|null} info - What the surface is drawing, or null while there is no picture.
 * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
 */
function pictureClaim(info) {
  if (!info) {
    return { icon: ICON.WARNING, detail: "there is no picture here to measure against" };
  }
  // The filter is a lookup table and nothing else, so on the test card there is nothing to
  // approximate: the card is generated at 1:1 with the pixels the filter is run over, the table
  // is applied to those bytes, and the result is written back without resampling. Over the whole
  // card and all 256 levels that was measured at 0 of 255 against the node itself.
  if (info.kind === BACKDROP_KIND.CARD) {
    return {
      icon: ICON.EXACT,
      detail: "0 of 255 over all 256 levels, and the picture's size only sets the frequency cap",
    };
  }
  // A picture the node published is held at 512 on its longest edge and then fitted into the
  // panel, and this table is applied to whatever that leaves. Reducing and then mapping is not
  // mapping and then reducing, and for a wave that wraps at black the two part company hard: on
  // the card fitted to a default node width they differ by up to 255 of 255, mean 30, across
  // more than half the picture. What survives is the flat areas, where a reduction keeps the
  // level it started from, so the hover claims those and nothing else.
  if (info.scale > REDUCED_AT) {
    return {
      icon: ICON.WARNING,
      detail: "flat areas exact, fine detail bands differently, up to 255 of 255",
    };
  }
  // A picture smaller than the panel is drawn larger than it was published, and the levels the
  // interpolation invents were never in the image. A wave that wraps at black turns an invented
  // level between two bands into a pixel at the far end of the range, so the magnified case gets
  // its own words rather than the copy's below: measured against the node at the shipped
  // defaults it is 229 of 255 at worst over 42 per cent of the channels.
  if (info.scale < MAGNIFIED_AT) {
    return {
      icon: ICON.WARNING,
      detail: "the picture is drawn larger, so its edges band differently, up to 229 of 255",
    };
  }
  // A picture drawn at the size it was published carries those pixels as they are, which is a
  // copy of the source rather than the source, so the hover says that rather than claiming a
  // bound it has not measured there.
  return { icon: ICON.EXACT, detail: "the curve is exact, this picture is a copy" };
}

/**
 * The claim for the surface's glyph, with what the run reads leading it where that differs.
 *
 * @param {object|null} info - What the surface is drawing, or null while there is no picture.
 * @param {object} model - Model from `readModel`.
 * @returns {{icon: string, detail: string, note?: string}} The glyph, its hover text, and the
 *   state that belongs on screen where there is one.
 */
function fidelityText(info, model) {
  const claim = pictureClaim(info);
  // A linked input is the one of these somebody can act on, so it is the one with words on
  // screen. It leads the hover as well, and the measurement follows it rather than being
  // replaced by it, since the hover has room for both and the bound is reachable nowhere else.
  const linked = linkedNote(model.linked);
  if (!linked) return claim;
  return {
    icon: ICON.WARNING,
    detail: `a linked input is read off the link, so the run reads another value. ${claim.detail}`,
    note: linked,
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
 * @param {{refresh: () => void}} surface - Surface from `createFilterSurface`.
 * @returns {() => void} Unhooks the listener.
 */
function watchRuns(node, surface) {
  return onNodeDone(node, () => {
    try {
      surface.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to ask for the image again:`, error);
    }
  });
}

/**
 * Read whether the interface is drawn at all.
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
 * Append the preview to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachNovaFilter(node) {
  for (const name of WIDGETS) {
    if (!findWidget(node, name)) return;
  }

  // Width of the image the node last received, and the card's own while there is none. The
  // filter and the curve read the frequency cap through this one number so they can never
  // disagree about what the node would do, which holds only while a change to it repaints
  // both. The curve is built after the surface and the backdrop is read a microtask later, so
  // the surface is given a call that finds the curve rather than the curve itself.
  const source = { width: 0 };
  let curve = null;
  // The width the node published on its last run gives the same cap the node applies. While no
  // image has arrived the cap comes from the test card's own 288, which is 144, and the widget
  // stops at 100, so nothing is capped on the card; a narrower image caps the node and this
  // preview alike.
  const model = () => readModel(node, source.width > 0 ? source.width : TEST_CARD.width);

  const surface = createFilterSurface({
    node,
    backdrop: trackedBackdrop(node, source, () => curve?.schedulePaint()),
    filter: (input, output) => {
      const lut = model().lut;
      const from = input.data;
      const to = output.data;
      // Alpha is carried through by the copy the surface makes. The node reads an RGB image,
      // so the table reaches three channels and never the fourth.
      for (let offset = 0; offset < from.length; offset += 4) {
        to[offset] = lut[from[offset]];
        to[offset + 1] = lut[from[offset + 1]];
        to[offset + 2] = lut[from[offset + 2]];
      }
    },
    fidelity: (info) => fidelityText(info, model()),
    height: SURFACE_HEIGHT,
    metrics: true,
  });

  // The curve is drawn as well as the picture. The wrap at black is the character of this
  // filter: two very different input brightnesses land on the same output, and anything the
  // wave carries below black comes back as white.
  curve = createNovaCurve(node, model);

  // Both appended after every schema widget and never inserted: `serialize` writes
  // `widgets_values` by absolute index while `configure` reads it with a compacted counter, so a
  // widget placed before a serialising one loads every later value into the wrong widget. Both
  // serialize flags are set by the two helpers.
  appendFilterWidget(node, surface);
  appendInterfaceWidget(node, curve, { name: CURVE_WIDGET_NAME, type: CURVE_WIDGET_TYPE });

  const repaint = () => {
    surface.schedulePaint();
    curve.schedulePaint();
  };

  for (const name of WIDGETS) chainWidgetCallback(node, name, repaint);

  const stopWatchingRuns = watchRuns(node, surface);

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link changes
  // no widget value, so the callbacks above never hear about it.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      repaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      repaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widgets registered and
  // their elements in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      stopWatchingRuns();
      curve.dispose();
      surface.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the nova preview:`, error);
    }
    return result;
  };

  repaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Nova Filter", "Nova preview"],
      name: "Show the nova preview",
      tooltip:
        "Draw the before and after picture and the tone curve under the widgets of Image " +
        "Nova Filter. The widgets themselves are always available. This applies to nodes " +
        "added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise wrap
    // the prototype a second time and append a second preview.
    if (proto.__was_nova_filter_wrapped) return;
    proto.__was_nova_filter_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachNovaFilter(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the nova preview:`, error);
      }
      return result;
    };
  },
});
