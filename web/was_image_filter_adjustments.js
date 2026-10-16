/**
 * Before and after preview for the Image Filter Adjustments node.
 *
 * The eight widgets compound in a fixed order over the buffer the surface hands across, bytes
 * and RGBA with no margin.
 */

import { app } from "../../scripts/app.js";
import {
  BACKDROP_KIND,
  appendFilterWidget,
  createFilterSurface,
  formatFactor,
} from "./interface/filter_surface.js";
import { ICON } from "./interface/icons.js";
import { roundHalfEven } from "./interface/python_arithmetic.js";
import { onNodeDone } from "./interface/run_events.js";
import { chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.FilterAdjustmentsUI";
const NODE_NAME = "Image Filter Adjustments";
const SETTING_ID = "WAS.FilterAdjustments.ShowInterface";

const BRIGHTNESS = "brightness";
const CONTRAST = "contrast";
const SATURATION = "saturation";
const SHARPNESS = "sharpness";
const BLUR = "blur";
const GAUSSIAN_BLUR = "gaussian_blur";
const EDGE_ENHANCE = "edge_enhance";
const DETAIL_ENHANCE = "detail_enhance";

// Every widget the preview reads, in the order the node applies them. None of them reads off
// its own number: a saturation of 1.8 and a sharpness of 1.8 are the same text and nothing
// alike, and the two blurs, the edge enhancement and the detail filter all compound on each
// other, so the order is the whole of what the preview has to show.
const WIDGETS = [
  BRIGHTNESS,
  CONTRAST,
  SATURATION,
  SHARPNESS,
  BLUR,
  GAUSSIAN_BLUR,
  EDGE_ENHANCE,
  DETAIL_ENHANCE,
];

// The schema's own defaults, read only when a widget cannot be.
const DEFAULTS = {
  [BRIGHTNESS]: 0.0,
  [CONTRAST]: 1.0,
  [SATURATION]: 1.0,
  [SHARPNESS]: 1.0,
  [BLUR]: 0,
  [GAUSSIAN_BLUR]: 0.0,
  [EDGE_ENHANCE]: 0.0,
};

// Height of the appended widget in node units. The picture is drawn twice side by side, so
// this is taller than an interface drawing a plot and no taller than one drawing one picture.
const UI_HEIGHT = 240;

// What the two panels are called.
const LABELS = { before: "input", after: "adjusted" };

// How many box passes per axis Pillow's Gaussian blur runs, and the fixed point it accumulates
// them in. The half is what makes the shift that follows round rather than truncate.
const BLUR_PASSES = 3;
const BLUR_ONE = 1 << 24;
const BLUR_HALF = 1 << 23;

// The most the `blur` widget's passes are allowed to cost one repaint, in pixels times passes.
// The 5x5 kernel reads sixteen taps for each of three channels of every pixel, the widget asks
// for up to sixteen passes, and a repaint runs inside one animation frame with the whole graph
// waiting on it. Sixteen passes over a wide panel is most of a second of arithmetic inside that
// one frame. Measured at about 80 nanoseconds per pixel per pass, so this is a repaint of
// roughly fifty milliseconds: the passes that fit are drawn and the footer says how many
// whenever any are held back.
const BLUR_BUDGET = 600000;

/**
 * One of Pillow's built-in convolution kernels, transcribed and prepared for use.
 *
 * @param {number} size - Width and height of the kernel, 3 or 5.
 * @param {number} scale - What Pillow divides the taps by.
 * @param {number[]} taps - The kernel, row by row.
 * @returns {object} The kernel, ready to convolve with.
 */
function buildKernel(size, scale, taps) {
  const radius = (size - 1) / 2;
  const weights = new Float32Array(size * size);
  // Every filter used here declares an offset of zero, so nothing is added after the scale.
  for (let i = 0; i < taps.length; i++) weights[i] = Math.fround(taps[i] / scale);

  // Flat typed arrays rather than an array of objects. A 5x5 blur reads sixteen taps for each of
  // three channels of each pixel and the widget asks for up to sixteen passes of it, so a
  // property lookup per tap is the difference between a repaint and a stall.
  const dx = [];
  const dy = [];
  const kept = [];
  const starts = [0];
  for (let ky = 0; ky < size; ky++) {
    for (let kx = 0; kx < size; kx++) {
      const weight = weights[ky * size + kx];
      // Adding a zero product to a float32 accumulator changes nothing, so dropping the zero
      // taps is exact, and it leaves the 5x5 blur reading 16 pixels rather than 25.
      if (weight === 0) continue;
      // Pillow reads the kernel's first row against the row below the pixel. Every kernel here
      // is symmetric under that flip, and the order the rows are accumulated in is not.
      dy.push(radius - ky);
      dx.push((kx - radius) * 4);
      kept.push(weight);
    }
    if (kept.length > starts[starts.length - 1]) starts.push(kept.length);
  }

  return {
    size,
    radius,
    dx: Int32Array.from(dx),
    dy: Int32Array.from(dy),
    weights: Float32Array.from(kept),
    starts: Int32Array.from(starts),
    offsets: new Int32Array(kept.length),
    stride: 0,
  };
}

/**
 * Where each of a kernel's taps sits relative to the pixel, for one buffer width.
 *
 * @param {object} kernel - Kernel from `buildKernel`.
 * @param {number} stride - Bytes per row of the buffer.
 * @returns {Int32Array} The offset of every tap, row by row.
 */
function kernelOffsets(kernel, stride) {
  if (kernel.stride === stride) return kernel.offsets;
  for (let i = 0; i < kernel.offsets.length; i++) {
    kernel.offsets[i] = kernel.dy[i] * stride + kernel.dx[i];
  }
  kernel.stride = stride;
  return kernel.offsets;
}

// The four filters the node reaches for, transcribed from `PIL.ImageFilter`.
const SMOOTH = buildKernel(3, 13, [1, 1, 1, 1, 5, 1, 1, 1, 1]);
const BOX_BLUR = buildKernel(5, 16, [
  1, 1, 1, 1, 1,
  1, 0, 0, 0, 1,
  1, 0, 0, 0, 1,
  1, 0, 0, 0, 1,
  1, 1, 1, 1, 1,
]);
const EDGE_ENHANCE_MORE = buildKernel(3, 1, [-1, -1, -1, -1, 9, -1, -1, -1, -1]);
const DETAIL = buildKernel(3, 6, [0, -1, 0, -1, 10, -1, 0, -1, 0]);

// What Pillow's convolution starts its accumulator at. The half is what makes the truncation
// that follows round, and it is seeded rather than added at the end: the two differ on the
// sums that land within an ulp of a half, which is about one byte in twenty thousand.
const KERNEL_SEED = 0.5;

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
 * Test whether one of a node's inputs is filled by a link.
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
 * Read every setting the node applies, as the node reads them.
 *
 * @param {object} node - Node the preview is drawn on.
 * @returns {object} The eight settings, plus the names of any that a link fills in.
 */
function readSettings(node) {
  return {
    brightness: widgetNumber(node, BRIGHTNESS),
    contrast: widgetNumber(node, CONTRAST),
    saturation: widgetNumber(node, SATURATION),
    sharpness: widgetNumber(node, SHARPNESS),
    blur: Math.trunc(widgetNumber(node, BLUR)),
    gaussianBlur: widgetNumber(node, GAUSSIAN_BLUR),
    edgeEnhance: widgetNumber(node, EDGE_ENHANCE),
    detail: String(findWidget(node, DETAIL_ENHANCE)?.value ?? "false") === "true",
    linked: WIDGETS.filter((name) => inputLinked(node, name)),
  };
}

/**
 * How many of the `blur` widget's passes a repaint can afford over a picture of this size.
 *
 * @param {number} asked - Passes the widget holds.
 * @param {number} width - Buffer width in pixels.
 * @param {number} height - Buffer height in pixels.
 * @returns {number} The passes to draw, never fewer than one where any were asked for.
 */
function blurPasses(asked, width, height) {
  if (!(asked > 0)) return 0;
  const pixels = Math.max(1, width * height);
  // A preview that freezes the graph for half a second on every step of a slider is not usable,
  // and one that quietly draws a different amount of blur than the node will is not honest, so
  // the count is bounded and the footer states it.
  return Math.max(1, Math.min(asked, Math.floor(BLUR_BUDGET / pixels)));
}

/**
 * Whether a setting is one of the four cut to a fixed shape in pixels.
 *
 * @param {object} settings - Settings from `readSettings`.
 * @returns {boolean} True while any of the four is doing something.
 */
function usesFixedKernels(settings) {
  // Nothing can be scaled about them: a 5x5 kernel has no fractional size, so the answer is
  // which of the four is running rather than by how much.
  return (
    settings.sharpness !== 1
    || settings.blur > 0
    || settings.edgeEnhance > 0
    || settings.detail
  );
}

/**
 * Whether a setting reaches past the pixel it is applied to.
 *
 * @param {object} settings - Settings from `readSettings`.
 * @returns {boolean} True while any neighbourhood step is doing something.
 */
function usesNeighbours(settings) {
  return usesFixedKernels(settings) || settings.gaussianBlur > 0;
}

/**
 * Whether any setting is doing anything at all.
 *
 * @param {object} settings - Settings from `readSettings`.
 * @returns {boolean} True while at least one step would run.
 */
function usesAnything(settings) {
  return (
    settings.brightness !== 0
    || settings.contrast !== 1
    || settings.saturation !== 1
    || usesNeighbours(settings)
  );
}

const scratch = [];

/**
 * A working buffer of a given length, reused between repaints.
 *
 * @param {number} slot - Which buffer to answer.
 * @param {number} length - Length in bytes.
 * @returns {Uint8ClampedArray} The buffer.
 */
function workspace(slot, length) {
  // Neither a convolution nor a box pass can be run in place, and a preview repaints on every
  // step of a widget, so a buffer allocated per pass would hand the collector a picture per
  // frame. One buffer per slot, kept until the picture changes size.
  if (!scratch[slot] || scratch[slot].length !== length) {
    scratch[slot] = new Uint8ClampedArray(length);
  }
  return scratch[slot];
}

/**
 * Convolve one buffer into another, exactly as `ImagingFilter` does.
 *
 * @param {Uint8ClampedArray} src - Source pixels, RGBA.
 * @param {Uint8ClampedArray} dst - Destination pixels, RGBA, overwritten.
 * @param {number} width - Width in pixels.
 * @param {number} height - Height in pixels.
 * @param {object} kernel - Kernel from `buildKernel`.
 * @returns {void}
 */
function convolve(src, dst, width, height, kernel) {
  // Filling the destination from the source first is the border Pillow copies rather than
  // computes: one pixel for a 3x3 kernel and two for a 5x5, on all four sides.
  dst.set(src);
  const radius = kernel.radius;
  // An image too small to hold the kernel is copied whole, as Pillow copies it.
  if (width < kernel.size || height < kernel.size) return;

  const stride = width * 4;
  const offsets = kernelOffsets(kernel, stride);
  const weights = kernel.weights;
  const starts = kernel.starts;
  const rowCount = starts.length - 1;
  const fround = Math.fround;

  // Nothing below is an equivalent formulation of the sum. The orders that look equivalent
  // disagree with Pillow on about one byte in twenty thousand, and this is the one that was
  // measured to agree on all of them.
  for (let y = radius; y < height - radius; y++) {
    const line = y * stride;
    for (let x = radius; x < width - radius; x++) {
      const base = line + x * 4;
      // The three channels are accumulated together rather than one after another. Each one
      // is its own float32 chain in Pillow's own order, and nothing crosses between them, so
      // the answer is the same to the byte; what is saved is two thirds of the tap lookups.
      let red = KERNEL_SEED;
      let green = KERNEL_SEED;
      let blue = KERNEL_SEED;
      for (let r = 0; r < rowCount; r++) {
        const end = starts[r + 1];
        let t = starts[r];
        let offset = offsets[t];
        let weight = weights[t];
        let accRed = fround(src[base + offset] * weight);
        let accGreen = fround(src[base + offset + 1] * weight);
        let accBlue = fround(src[base + offset + 2] * weight);
        for (t++; t < end; t++) {
          offset = offsets[t];
          weight = weights[t];
          accRed = fround(accRed + fround(src[base + offset] * weight));
          accGreen = fround(accGreen + fround(src[base + offset + 1] * weight));
          accBlue = fround(accBlue + fround(src[base + offset + 2] * weight));
        }
        red = fround(red + accRed);
        green = fround(green + accGreen);
        blue = fround(blue + accBlue);
      }
      dst[base] = red <= 0 ? 0 : red >= 255 ? 255 : Math.trunc(red);
      dst[base + 1] = green <= 0 ? 0 : green >= 255 ? 255 : Math.trunc(green);
      dst[base + 2] = blue <= 0 ? 0 : blue >= 255 ? 255 : Math.trunc(blue);
    }
  }
}

/**
 * Blend two buffers as `Image.blend` does, into the second of them.
 *
 * @param {Uint8ClampedArray} degenerate - The copy the enhancement measures from.
 * @param {Uint8ClampedArray} image - The picture, overwritten with the result.
 * @param {number} factor - The enhancement factor.
 * @returns {void}
 */
function blendInto(degenerate, image, factor) {
  const f = Math.fround(factor);
  // Every `ImageEnhance` is this against a degenerate copy of the image, so a factor above 1
  // extrapolates away from that copy and a negative factor extrapolates past it, which is what
  // a negative sharpness draws.
  for (let i = 0; i < image.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const a = degenerate[i + channel];
      const value = Math.fround(a + Math.fround(f * (image[i + channel] - a)));
      image[i + channel] = value <= 0 ? 0 : value >= 255 ? 255 : Math.trunc(value);
    }
  }
}

/**
 * Apply brightness and then contrast, as the node applies them to the tensor.
 *
 * @param {Uint8ClampedArray} data - Pixels, RGBA, overwritten.
 * @param {number} brightness - Amount added to every sample.
 * @param {number} contrast - Multiplier applied to every sample.
 * @returns {void}
 */
function applyTone(data, brightness, contrast) {
  // Both are tensor operations, so every step runs through `Math.fround`. A byte to float32 and
  // back is lossless for all 256 levels, which is what lets the preview hold bytes throughout.
  const add = Math.fround(brightness);
  const multiply = Math.fround(contrast);
  const table = new Uint8Array(256);

  for (let level = 0; level < 256; level++) {
    let value = Math.fround(level / 255);
    if (brightness !== 0) {
      value = Math.fround(value + add);
      value = value < 0 ? 0 : value > 1 ? 1 : value;
    }
    if (contrast !== 1) {
      value = Math.fround(value * multiply);
      value = value < 0 ? 0 : value > 1 ? 1 : value;
    }
    table[level] = clamp(Math.trunc(Math.fround(255 * value)), 0, 255);
  }

  for (let i = 0; i < data.length; i += 4) {
    data[i] = table[data[i]];
    data[i + 1] = table[data[i + 1]];
    data[i + 2] = table[data[i + 2]];
  }
}

/**
 * Apply saturation, which is `ImageEnhance.Color`.
 *
 * @param {Uint8ClampedArray} data - Pixels, RGBA, overwritten.
 * @param {number} factor - Colour strength.
 * @returns {void}
 */
function applySaturation(data, factor) {
  const f = Math.fround(factor);
  for (let i = 0; i < data.length; i += 4) {
    // `ImageEnhance.Color` blends against the image's own greyscale, and Pillow's greyscale is
    // this integer luma, `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16`, rather than any of the
    // several a browser offers.
    const grey = (data[i] * 19595 + data[i + 1] * 38470 + data[i + 2] * 7471 + 0x8000) >>> 16;
    for (let channel = 0; channel < 3; channel++) {
      const value = Math.fround(grey + Math.fround(f * (data[i + channel] - grey)));
      data[i + channel] = value <= 0 ? 0 : value >= 255 ? 255 : Math.trunc(value);
    }
  }
}

/**
 * Composite an edge enhanced copy over the picture through a constant mask.
 *
 * @param {Uint8ClampedArray} overlay - The enhanced copy, which the mask lets through.
 * @param {Uint8ClampedArray} data - The picture, overwritten with the result.
 * @param {number} mask - Mask level, 0 to 255.
 * @returns {void}
 */
function compositeThrough(overlay, data, mask) {
  for (let i = 0; i < data.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const base = data[i + channel];
      data[i + channel] = Math.floor(base + ((overlay[i + channel] - base) * mask) / 255 + 0.5);
    }
  }
}

/**
 * The box radius three passes need to stand in for a Gaussian of a given radius.
 *
 * @param {number} radius - The radius the widget holds.
 * @returns {number} The box radius, with its fractional part.
 */
function boxRadius(radius) {
  // Everything here is held to float32, matching the C, and the answer decides both the
  // integer radius and the weights.
  const f = Math.fround;
  const sigma2 = f(f(f(radius) * f(radius)) / BLUR_PASSES);
  const length = f(Math.sqrt(12.0 * sigma2 + 1.0));
  const whole = f(Math.floor((length - 1.0) / 2.0));
  let part = f((2 * whole + 1) * f(whole * (whole + 1) - 3 * sigma2));
  part = f(part / f(6 * f(sigma2 - (whole + 1) * (whole + 1))));
  return f(whole + part);
}

/**
 * One horizontal box pass, in the 24 bit fixed point Pillow accumulates it in.
 *
 * @param {Uint8ClampedArray} src - Source pixels, RGBA.
 * @param {Uint8ClampedArray} dst - Destination pixels, RGBA, overwritten.
 * @param {number} width - Width in pixels.
 * @param {number} height - Height in pixels.
 * @param {number} radius - The box radius, with its fractional part.
 * @returns {void}
 */
function boxPass(src, dst, width, height, radius) {
  const whole = Math.trunc(radius);
  // The weights are truncated to whole 24 bit fractions exactly as the C truncates them, so the
  // rounding at the end is Pillow's rather than an equivalent of it.
  const weight = Math.trunc(Math.fround(BLUR_ONE / Math.fround(radius * 2 + 1)));
  const edgeWeight = Math.trunc((BLUR_ONE - (whole * 2 + 1) * weight) / 2);
  const last = width - 1;
  const stride = width * 4;

  for (let y = 0; y < height; y++) {
    const line = y * stride;
    // Alpha is blurred with the colours, which is what Pillow does to every band of an RGBA
    // image, so a cut-out spreads outward rather than keeping the edge it started with.
    for (let channel = 0; channel < 4; channel++) {
      // Reading past either end of the row reads the edge pixel, the clamp `ImagingBoxBlur`
      // does, and no margin is kept around the picture.
      const at = (x) => src[line + (x < 0 ? 0 : x > last ? last : x) * 4 + channel];
      // The total for the pixel before the first one, with the edge repeated outward.
      let total = at(0) * (whole + 1);
      for (let x = 0; x < whole; x++) total += at(x);
      for (let x = 0; x <= last; x++) {
        const behind = at(x - whole - 1);
        total += at(x + whole) - behind;
        const bulk = total * weight + (behind + at(x + whole + 1)) * edgeWeight;
        dst[line + x * 4 + channel] = (bulk + BLUR_HALF) >>> 24;
      }
    }
  }
}

/**
 * Turn a buffer on its side, so the vertical passes are the horizontal ones again.
 *
 * @param {Uint8ClampedArray} src - Source pixels, RGBA.
 * @param {Uint8ClampedArray} dst - Destination pixels, RGBA, overwritten.
 * @param {number} width - Source width in pixels.
 * @param {number} height - Source height in pixels.
 * @returns {void}
 */
function transpose(src, dst, width, height) {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const from = (y * width + x) * 4;
      const to = (x * height + y) * 4;
      dst[to] = src[from];
      dst[to + 1] = src[from + 1];
      dst[to + 2] = src[from + 2];
      dst[to + 3] = src[from + 3];
    }
  }
}

/**
 * Blur a buffer exactly as `ImageFilter.GaussianBlur` blurs an image.
 *
 * @param {Uint8ClampedArray} data - Pixels, RGBA, overwritten.
 * @param {number} width - Width in pixels.
 * @param {number} height - Height in pixels.
 * @param {number} radius - Radius in pixels of this picture.
 * @returns {void}
 */
function gaussianBlur(data, width, height, radius) {
  // The browser's canvas blur is the Filter Effects rule and samples transparent black outside
  // the source, which is where the 40 of 255 comes from. Pillow's algorithm ports whole instead:
  // it needs no canvas, no margin around the picture and no approximation, and it reproduces
  // every radius from 0.1 to the widget's own maximum byte for byte.
  if (!(radius > 0) || width < 1 || height < 1) return;
  const box = boxRadius(radius);
  if (!Number.isFinite(box) || box < 0) return;

  let source = workspace(1, data.length);
  let target = workspace(2, data.length);
  source.set(data);

  for (let pass = 0; pass < BLUR_PASSES; pass++) {
    boxPass(source, target, width, height, box);
    const swap = source;
    source = target;
    target = swap;
  }
  transpose(source, target, width, height);
  const swap = source;
  source = target;
  target = swap;
  for (let pass = 0; pass < BLUR_PASSES; pass++) {
    boxPass(source, target, height, width, box);
    const other = source;
    source = target;
    target = other;
  }
  transpose(source, target, height, width);
  data.set(target);
}

/**
 * Run every step the node runs, in the node's order, over the buffer the surface supplied.
 *
 * @param {ImageData} source - The picture the surface handed over. Read only through the copy
 *   in `target`.
 * @param {ImageData} target - A copy of it, overwritten with the result.
 * @param {object} info - What the picture is, from the surface.
 * @param {object} settings - Settings from `readSettings`.
 * @returns {void}
 */
function applyAdjustments(source, target, info, settings) {
  const width = target.width;
  const height = target.height;
  const data = target.data;
  const length = data.length;

  if (settings.brightness !== 0 || settings.contrast !== 1) {
    applyTone(data, settings.brightness, settings.contrast);
  }

  if (settings.saturation !== 1) {
    applySaturation(data, settings.saturation);
  }

  if (settings.sharpness !== 1) {
    // `ImageEnhance.Sharpness` is the same truncating blend as saturation, against a `SMOOTH`
    // pass rather than a greyscale.
    const smoothed = workspace(0, length);
    convolve(data, smoothed, width, height, SMOOTH);
    blendInto(smoothed, data, settings.sharpness);
  }

  if (settings.blur > 0) {
    // A pass cannot be run in place, so the two buffers change places between passes rather
    // than the result being copied back over the picture each time, and the copy is made only
    // where an odd number of passes leaves the result in the scratch.
    const passes = blurPasses(settings.blur, width, height);
    let from = data;
    let into = workspace(0, length);
    for (let pass = 0; pass < passes; pass++) {
      convolve(from, into, width, height, BOX_BLUR);
      const swap = from;
      from = into;
      into = swap;
    }
    if (from !== data) data.set(from);
  }

  if (settings.gaussianBlur > 0) {
    // The radius is in pixels of the image the node was given. On the test card that is this
    // picture, and on a published thumbnail it is a picture this one was reduced from.
    gaussianBlur(data, width, height, settings.gaussianBlur / (info.scale || 1));
  }

  if (settings.edgeEnhance > 0) {
    const edges = workspace(0, length);
    convolve(data, edges, width, height, EDGE_ENHANCE_MORE);
    // The whole step is blended through this one level, and the widget steps by 0.01, so 0.3 and
    // 0.7 both land exactly on half a byte: 76 and 178 under the node's rule, 77 and 179 under
    // `Math.round`.
    compositeThrough(edges, data, clamp(roundHalfEven(settings.edgeEnhance * 255), 0, 255));
  }

  if (settings.detail) {
    const detailed = workspace(0, length);
    convolve(data, detailed, width, height, DETAIL);
    data.set(detailed);
  }
}

/**
 * What this preview does not reproduce, measured from the settings and the picture on screen.
 *
 * The claim changes with the settings, as the fidelity does.
 *
 * @param {object} settings - Settings from `readSettings`.
 * @param {object|null} info - What the picture is, from the surface.
 * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
 */
function measuredClaim(settings, info) {
  // A step that is skipped is skipped in both places, so a node with nothing set draws the
  // picture it was given whatever the picture is.
  if (!usesAnything(settings)) return { icon: ICON.EXACT, detail: "nothing is set yet" };

  // A blur held back by the cost budget is tested before the reduction below and outranks it,
  // since nothing else carries the count of passes.
  if (info && settings.blur > 0) {
    const passes = blurPasses(
      settings.blur,
      info.width + info.margin * 2,
      info.height + info.margin * 2,
    );
    if (passes < settings.blur) {
      return {
        icon: ICON.WARNING,
        detail: `blur drawn at ${passes} of ${settings.blur} passes, more would stall the graph`,
      };
    }
  }

  // On anything but the test card nothing can be exact: the picture has already been reduced to
  // fit inside the node. What is said is not that it has fewer pixels, which is what a picture
  // inside a node always has, but what the reduction does to the steps that are running, which is
  // the part that changes as the settings do.
  if (info && info.kind !== BACKDROP_KIND.CARD) {
    const factor = formatFactor(info.scale);
    // The four fixed kernels are cut to a shape in pixels of the picture on screen rather than of
    // the image the node was given, so each of them reaches that factor further into the picture
    // than the render will.
    if (usesFixedKernels(settings)) {
      return {
        icon: ICON.WARNING,
        detail: `the softening reaches ${factor}x further than it will on the render`,
      };
    }
    // `gaussian_blur` is the one neighbourhood step scaled by the factor, so its reach is the
    // reach the render has and only the resampling stands between the two.
    if (settings.gaussianBlur > 0) {
      return {
        icon: ICON.APPROXIMATE,
        detail: `the radius is scaled ${factor}x with the picture, so this is close`,
      };
    }
    return {
      icon: ICON.APPROXIMATE,
      detail: "every step here is per pixel, so this is close rather than exact",
    };
  }

  // The test card is generated at the size it is drawn at, so every step reproduces the node byte
  // for byte.
  return { icon: ICON.EXACT, detail: "byte for byte what the node renders" };
}

/**
 * The claim for the surface's glyph, with what the run reads leading it where that differs.
 *
 * @param {object} node - Node the preview is drawn on.
 * @param {object|null} info - What the picture is, from the surface.
 * @returns {{icon: string, detail: string, note?: string}} The glyph, its hover text, and the
 *   state that belongs on screen where there is one.
 */
function fidelityNote(node, info) {
  const settings = readSettings(node);
  const claim = measuredClaim(settings, info);
  if (settings.linked.length === 0) return claim;
  // A preview drawn from a widget a link has replaced shows a picture the run never produces,
  // which is worth more than any rounding rule. It is also the one of these that changes what
  // somebody should do next, so it is the one with words on screen. It leads the hover as well,
  // and the measurement follows it rather than being replaced by it, since the hover has room for
  // both and the passes and the reach are reachable nowhere else.
  return {
    icon: ICON.WARNING,
    detail: `a linked setting is read off the link, so this is not what will run. ${claim.detail}`,
    note: settings.linked.length === 1
      ? `${settings.linked[0]} is linked`
      : `${settings.linked.length} settings are linked`,
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
 * Read whether the preview is drawn at all.
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
function attachPreview(node) {
  for (const name of WIDGETS) {
    if (!findWidget(node, name)) return;
  }

  // No margin is asked for. Every step here clamps at the edge of the buffer exactly where
  // Pillow clamps at the edge of the image, so the buffer being the visible picture is what
  // makes the border right rather than what makes it wrong.
  const surface = createFilterSurface({
    node,
    filter: (source, target, info) => applyAdjustments(source, target, info, readSettings(node)),
    fidelity: (info) => fidelityNote(node, info),
    labels: LABELS,
    height: UI_HEIGHT,
    metrics: true,
  });

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendFilterWidget` is for.
  appendFilterWidget(node, surface);

  for (const name of WIDGETS) {
    chainWidgetCallback(node, name, surface.schedulePaint, EXT_NAME);
  }

  const stopWatchingRuns = watchRuns(node, surface);

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link changes
  // no widget value, so the callbacks above never hear about it.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      surface.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      surface.schedulePaint();
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
      stopWatchingRuns();
      surface.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the preview:`, error);
    }
    return result;
  };

  surface.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Filter Adjustments", "Preview"],
      name: "Show the adjustments preview",
      tooltip:
        "Draw the input and the adjusted result under the widgets of Image Filter "
        + "Adjustments. The widgets themselves are always available. This applies to nodes "
        + "added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second preview.
    if (proto.__was_filter_adjustments_wrapped) return;
    proto.__was_filter_adjustments_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachPreview(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the preview:`, error);
      }
      return result;
    };
  },
});

// Exported so the arithmetic can be run against the node's own output rather than against a
// copy of it. ComfyUI imports this file for its one side effect and reads no export, and an
// export is inert, so nothing here depends on anyone taking them.
export { applyAdjustments, boxRadius, convolve, gaussianBlur, readSettings };
