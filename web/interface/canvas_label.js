/**
 * A short label drawn on a canvas, on a plate in the panel's own colours.
 *
 * The text keeps the theme's foreground; a swatch beside it carries the colour of whatever it names.
 */

/**
 * Draw a label on a plate, held inside the frame.
 *
 * @param {CanvasRenderingContext2D} ctx - The panel's context.
 * @param {{width: number, height: number, top: number, theme: object}} frame - The area the label
 *   stays inside, and the theme tokens it is drawn in.
 * @param {string} text - What it says.
 * @param {number} x - Left edge, or right edge with `align: "right"`.
 * @param {number} y - Vertical middle.
 * @param {{align?: string, swatch?: string, muted?: boolean}} [style] - Right alignment, a colour
 *   swatch before the text, and the muted text colour.
 * @returns {{x: number, y: number, width: number, height: number}} Where the plate landed.
 */
export function drawLabel(ctx, frame, text, x, y, style = {}) {
  const theme = frame.theme;
  ctx.save();
  ctx.font = "10px system-ui,sans-serif";
  ctx.textBaseline = "middle";
  const chip = style.swatch ? 9 : 0;
  const width = Math.ceil(ctx.measureText(text).width) + 8 + chip;
  const height = 14;
  let left = style.align === "right" ? x - width : x;
  left = Math.max(1, Math.min(frame.width - width - 1, left));
  const top = Math.max(frame.top + 1, Math.min(frame.height - height - 1, y - height / 2));
  ctx.fillStyle = theme.bgDark ?? theme.bg;
  ctx.globalAlpha = 0.92;
  ctx.fillRect(left, top, width, height);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = theme.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(left + 0.5, top + 0.5, width - 1, height - 1);
  if (style.swatch) {
    ctx.fillStyle = style.swatch;
    ctx.fillRect(left + 4, top + 4, 5, height - 8);
  }
  ctx.fillStyle = style.muted ? theme.fgMuted : theme.fg;
  ctx.fillText(text, left + 4 + chip, top + height / 2 + 0.5);
  ctx.restore();
  return { x: left, y: top, width, height };
}
