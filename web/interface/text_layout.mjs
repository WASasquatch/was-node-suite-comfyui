/**
 * Where a block of text sits in the image a node draws on.
 *
 * Every number in and out is a pixel of that image, never a pixel on screen. Rounding is
 * Python's, and text is measured through the canvas.
 */

import { roundHalfEven, truncate } from "./python_arithmetic.mjs";

/**
 * Where a block sits inside the image, as `[x, y]` fractions of the space left over. `[0, 0]` is
 * the top left corner.
 */
export const ANCHORS = {
  "top left": [0.0, 0.0],
  "top center": [0.5, 0.0],
  "top right": [1.0, 0.0],
  "middle left": [0.0, 0.5],
  "middle center": [0.5, 0.5],
  "middle right": [1.0, 0.5],
  "bottom left": [0.0, 1.0],
  "bottom center": [0.5, 1.0],
  "bottom right": [1.0, 1.0],
};

/** How the lines sit inside the block, which is the node's `align` option. */
export const ALIGN = {
  LEFT: "left",
  CENTER: "center",
  RIGHT: "right",
};

// Taken off a measurement before it is rounded up.
const EPSILON = 1e-9;

/**
 * The width of one line as the node measures it, stroke included.
 *
 * The advance and the ink unioned, in whole pixels.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the node's font at its own size.
 * @param {string} line - The line to measure.
 * @param {number} [strokeWidth] - The node's `stroke_width`.
 * @returns {number} The width in image pixels.
 */
export function textWidth(ctx, line, strokeWidth = 0) {
  const metrics = ctx.measureText(line);
  const advance = Math.ceil(number(metrics.width) - EPSILON);
  // `actualBoundingBoxLeft` is positive where the ink reaches left of the pen.
  const inkLeft = -Math.ceil(number(metrics.actualBoundingBoxLeft) - EPSILON);
  const inkRight = Math.ceil(number(metrics.actualBoundingBoxRight) - EPSILON);
  return Math.max(advance, inkRight) - Math.min(0, inkLeft) + Math.max(0, strokeWidth) * 2;
}

/**
 * Break text into lines that fit a width.
 *
 * Line breaks are kept and each paragraph is wrapped on its own, at spaces.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the node's font at its own size.
 * @param {string} text - The text to lay out.
 * @param {number} maxWidth - Width to fit, in image pixels. 0 or less turns wrapping off.
 * @param {number} [strokeWidth] - The node's `stroke_width`, which widens every glyph.
 * @returns {string[]} The lines, in order.
 */
export function wrapLines(ctx, text, maxWidth, strokeWidth = 0) {
  const paragraphs = String(text ?? "").split("\n");
  if (!(maxWidth > 0)) return paragraphs;

  const lines = [];
  for (const paragraph of paragraphs) {
    let current = "";
    for (const word of paragraph.split(" ")) {
      const candidate = current ? `${current} ${word}` : word;
      if (current && textWidth(ctx, candidate, strokeWidth) > maxWidth) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    lines.push(current);
  }
  return lines;
}

/**
 * Measure a laid-out block of text.
 *
 * The line height is the step from one baseline to the next.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the node's font at its own size.
 * @param {string[]} lines - The lines, already wrapped.
 * @param {{ascent: number, descent: number}} face - The face's metrics at that size, from
 *   `faceMetrics` in `fonts.mjs`.
 * @param {number} [lineSpacing] - The node's `line_spacing`.
 * @param {number} [strokeWidth] - The node's `stroke_width`.
 * @returns {{width: number, height: number, lineHeight: number}} The block in image pixels.
 */
export function textBlock(ctx, lines, face, lineSpacing = 1, strokeWidth = 0) {
  const stroke = Math.max(0, strokeWidth);
  const natural = number(face?.ascent) + number(face?.descent) + stroke * 2;
  const lineHeight = Math.max(1, truncate(roundHalfEven(natural * Math.max(0.1, lineSpacing))));
  let width = 0;
  for (const line of lines) width = Math.max(width, textWidth(ctx, line, stroke));
  return {
    width: truncate(roundHalfEven(width)),
    height: lineHeight * Math.max(1, lines.length),
    lineHeight,
  };
}

/**
 * Place a block inside the image at a named position.
 *
 * The answer is negative where the block is larger than the space.
 *
 * @param {string} position - A key of `ANCHORS`. An unknown name is `middle center`.
 * @param {number[]} canvas - `[width, height]` of the image, in pixels.
 * @param {number[]} block - `[width, height]` of what is being placed.
 * @param {number} [margin] - Pixels held back from every edge.
 * @returns {number[]} `[x, y]` of the block's top left corner, in image pixels.
 */
export function anchorOrigin(position, canvas, block, margin = 0) {
  const [fractionX, fractionY] = ANCHORS[position] ?? ANCHORS["middle center"];
  const freeX = canvas[0] - block[0] - margin * 2;
  const freeY = canvas[1] - block[1] - margin * 2;
  return [
    truncate(roundHalfEven(margin + freeX * fractionX)),
    truncate(roundHalfEven(margin + freeY * fractionY)),
  ];
}

/**
 * Where one line starts inside the block.
 *
 * @param {string} align - A value of `ALIGN`. Anything else is left aligned, as the node reads it.
 * @param {number} blockWidth - The block's width in image pixels.
 * @param {number} width - This line's width in image pixels.
 * @returns {number} How far right of the block's left edge the line's pen starts.
 */
export function lineStart(align, blockWidth, width) {
  if (align === ALIGN.CENTER) return (blockWidth - width) / 2;
  if (align === ALIGN.RIGHT) return blockWidth - width;
  return 0;
}

/**
 * The panel drawn behind the block.
 *
 * The panel is one pixel wider and one taller than the padding alone says.
 *
 * @param {number[]} origin - `[x, y]` of the block, offset applied, in image pixels.
 * @param {number[]} block - `[width, height]` of the block, in image pixels.
 * @param {number} padding - The node's `background_padding`.
 * @returns {{x: number, y: number, w: number, h: number}} The panel in image pixels.
 */
export function panelBox(origin, block, padding) {
  const pad = Math.max(0, padding);
  return {
    x: origin[0] - pad,
    y: origin[1] - pad,
    w: block[0] + pad * 2 + 1,
    h: block[1] + pad * 2 + 1,
  };
}

/**
 * The `lineWidth` a canvas needs to put the same outline outside a glyph as the node does.
 *
 * @param {number} strokeWidth - The node's `stroke_width`.
 * @returns {number} What to put in `ctx.lineWidth`, with round joins and caps.
 */
export function strokeLineWidth(strokeWidth) {
  // `strokeText` centres its width on the path, so half of it falls inside the glyph.
  return Math.max(0, strokeWidth) * 2;
}

/**
 * Everything `draw_text_layer` works out before it draws a pixel.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the node's font at its own size.
 * @param {object} options - The node's widgets, in the node's own names and units.
 * @param {string} options.text - The `text` widget.
 * @param {{ascent: number, descent: number}} options.face - The face's metrics at `font_size`.
 * @param {number[]} options.canvas - `[width, height]` of the image drawn on.
 * @param {string} [options.position] - A key of `ANCHORS`.
 * @param {string} [options.align] - A value of `ALIGN`.
 * @param {number[]} [options.offset] - `[x, y]` added to the anchored position.
 * @param {number} [options.margin] - Pixels held back from every edge.
 * @param {number} [options.lineSpacing] - Multiplier on the face's line height.
 * @param {number} [options.wrapWidth] - Width to wrap at, or 0 for no wrapping.
 * @param {number} [options.strokeWidth] - Outline width in pixels.
 * @param {number} [options.backgroundPadding] - How far the panel extends past the block.
 * @returns {{lines: string[], block: object, x: number, y: number, panel: object,
 *   rows: Array<{text: string, x: number, baseline: number}>}} The lines, the block's size, its top
 *   left corner, the panel behind it, and each line's pen position, all in image pixels. `baseline`
 *   is where the glyphs sit, which is the line's own top plus the face's ascent.
 */
export function layoutText(ctx, options) {
  const {
    text = "",
    face,
    canvas,
    position = "middle center",
    align = ALIGN.CENTER,
    offset = [0, 0],
    margin = 0,
    lineSpacing = 1,
    wrapWidth = 0,
    strokeWidth = 0,
    backgroundPadding = 0,
  } = options ?? {};

  const stroke = Math.max(0, strokeWidth);
  const lines = wrapLines(ctx, text, wrapWidth, stroke);
  const block = textBlock(ctx, lines, face, lineSpacing, stroke);
  const anchored = anchorOrigin(position, canvas, [block.width, block.height], margin);
  const x = anchored[0] + number(offset[0]);
  const y = anchored[1] + number(offset[1]);

  const rows = lines.map((line, index) => ({
    text: line,
    x: x + lineStart(align, block.width, textWidth(ctx, line, stroke)),
    baseline: y + index * block.lineHeight + number(face?.ascent),
  }));

  return {
    lines,
    block,
    x,
    y,
    panel: panelBox([x, y], [block.width, block.height], backgroundPadding),
    rows,
  };
}

/**
 * Read a number, answering zero for anything that is not one.
 *
 * @param {*} value - Value to read.
 * @returns {number} The number, or 0.
 */
function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}
